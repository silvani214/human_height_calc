import React, { useEffect, useRef, useState } from "react";
import { Dimensions, Platform, StyleSheet, Text, View } from "react-native";

import { Camera } from "expo-camera";

import * as posedetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
import {
  bundleResourceIO,
  cameraWithTensors,
} from "@tensorflow/tfjs-react-native";
import { CameraType } from "expo-camera/build/Camera.types";
import { ExpoWebGLRenderingContext } from "expo-gl";
import * as ScreenOrientation from "expo-screen-orientation";
import Svg, { Circle } from "react-native-svg";

// tslint:disable-next-line: variable-name
const TensorCamera = cameraWithTensors(Camera);

const IS_ANDROID = Platform.OS === "android";
const IS_IOS = Platform.OS === "ios";
const RATIO = 8.5;
// Camera preview size.
//
// From experiments, to render camera feed without distortion, 16:9 ratio
// should be used fo iOS devices and 4:3 ratio should be used for android
// devices.
//
// This might not cover all cases.
const CAM_PREVIEW_WIDTH = Dimensions.get("window").width;
const CAM_PREVIEW_HEIGHT = CAM_PREVIEW_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

// The score threshold for pose detection results.
const MIN_KEYPOINT_SCORE = 0.3;

// The size of the resized output from TensorCamera.
//
// For movenet, the size here doesn't matter too much because the model will
// preprocess the input (crop, resize, etc). For best result, use the size that
// doesn't distort the image.
const OUTPUT_TENSOR_WIDTH = 180;
const OUTPUT_TENSOR_HEIGHT = OUTPUT_TENSOR_WIDTH / (IS_IOS ? 9 / 16 : 3 / 4);

// Whether to auto-render TensorCamera preview.
const AUTO_RENDER = false;

// Whether to load model from app bundle (true) or through network (false).
const LOAD_MODEL_FROM_BUNDLE = false;

export default function App() {
  const cameraRef = useRef(null);
  const [tfReady, setTfReady] = React.useState(false);
  const [model, setModel] = useState<posedetection.PoseDetector>();
  const [poses, setPoses] = useState<posedetection.Pose[]>();
  const [fps, setFps] = useState(0);
  const [orientation, setOrientation] =
    useState<ScreenOrientation.Orientation>();
  const [cameraType, setCameraType] = useState<CameraType>(
    Camera.Constants.Type.front
  );
  // Use `useRef` so that changing it won't trigger a re-render.
  //
  // - null: unset (initial value).
  // - 0: animation frame/loop has been canceled.
  // - >0: animation frame has been scheduled.
  const rafId = useRef<number | null>(null);

  useEffect(() => {
    async function prepare() {
      rafId.current = null;

      // Set initial orientation.
      const curOrientation = await ScreenOrientation.getOrientationAsync();
      setOrientation(curOrientation);

      // Listens to orientation change.
      ScreenOrientation.addOrientationChangeListener((event) => {
        setOrientation(event.orientationInfo.orientation);
      });

      // Camera permission.
      await Camera.requestCameraPermissionsAsync();

      // Wait for tfjs to initialize the backend.
      await tf.ready();

      // Load movenet model.
      // https://github.com/tensorflow/tfjs-models/tree/master/pose-detection
      const movenetModelConfig: posedetection.MoveNetModelConfig = {
        modelType: posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        enableSmoothing: true,
      };
      if (LOAD_MODEL_FROM_BUNDLE) {
        const modelJson = require("./offline_model/model.json");
        const modelWeights1 = require("./offline_model/group1-shard1of2.bin");
        const modelWeights2 = require("./offline_model/group1-shard2of2.bin");
        movenetModelConfig.modelUrl = bundleResourceIO(modelJson, [
          modelWeights1,
          modelWeights2,
        ]);
      }
      const model = await posedetection.createDetector(
        posedetection.SupportedModels.MoveNet,
        movenetModelConfig
      );
      setModel(model);

      // Ready!
      setTfReady(true);
    }

    prepare();
  }, []);

  useEffect(() => {
    // Called when the app is unmounted.
    return () => {
      if (rafId.current != null && rafId.current !== 0) {
        cancelAnimationFrame(rafId.current);
        rafId.current = 0;
      }
    };
  }, []);

  const handleCameraStream = async (
    images: IterableIterator<tf.Tensor3D>,
    updatePreview: () => void,
    gl: ExpoWebGLRenderingContext
  ) => {
    const loop = async () => {
      // Get the tensor and run pose detection.
      const imageTensor = images.next().value as tf.Tensor3D;

      const startTs = Date.now();
      const poses = await model!.estimatePoses(
        imageTensor,
        undefined,
        Date.now()
      );
      const latency = Date.now() - startTs;
      setFps(Math.floor(1000 / latency));
      setPoses(poses);
      tf.dispose([imageTensor]);

      if (rafId.current === 0) {
        return;
      }

      // Render camera preview manually when autorender=false.
      if (!AUTO_RENDER) {
        updatePreview();
        gl.endFrameEXP();
      }

      rafId.current = requestAnimationFrame(loop);
    };

    loop();
  };

  const renderPose = () => {
    if (poses != null && poses.length > 0) {
      const keypoints = poses[0].keypoints
        .filter((k) => (k.score ?? 0) > MIN_KEYPOINT_SCORE)
        .map((k) => {
          // Flip horizontally on android or when using back camera on iOS.
          const flipX = IS_ANDROID || cameraType === Camera.Constants.Type.back;
          const x = flipX ? getOutputTensorWidth() - k.x : k.x;
          const y = k.y;
          const cx =
            (x / getOutputTensorWidth()) *
            (isPortrait() ? CAM_PREVIEW_WIDTH : CAM_PREVIEW_HEIGHT);
          const cy =
            (y / getOutputTensorHeight()) *
            (isPortrait() ? CAM_PREVIEW_HEIGHT : CAM_PREVIEW_WIDTH);
          return (
            <Circle
              key={`skeletonkp_${k.name}`}
              cx={cx}
              cy={cy}
              r="4"
              strokeWidth="2"
              fill="#00AA00"
              stroke="white"
            />
          );
        });

      return <Svg style={styles.svg}>{keypoints}</Svg>;
    } else {
      return <View></View>;
    }
  };

  const renderFps = () => {
    return (
      <View style={styles.fpsContainer}>
        <Text>FPS: {fps}</Text>
      </View>
    );
  };

  function arePointsApproximatelyInStraightLine(
    point1: any,
    point2: any,
    point3: any,
    min = 0.3,
    max = 3
  ) {
    // Slope formula: (y2 - y1) / (x2 - x1)
    const slope1 = (point2.y - point1.y) / (point2.x - point1.x);
    const slope2 = (point3.y - point2.y) / (point3.x - point2.x);

    // Check if slopes are approximately equal
    const slopeDifference = Math.abs(slope1 - slope2);

    if (slopeDifference <= min || slopeDifference >= max) console.log("...");
    else console.log(slopeDifference);

    return slopeDifference <= min || slopeDifference >= max;
  }

  const renderHeight = () => {
    let height = 0;
    let isStraight = false;
    let isDetected = false;
    let points: { [key: string]: posedetection.Keypoint } = {};

    //fetch pose features for body
    if (poses != null && poses.length > 0) {
      const keypoints = poses[0].keypoints;
      keypoints
        .filter((k) => (k.score ?? 0) > MIN_KEYPOINT_SCORE)
        .map((point) => {
          points[point.name] = { ...point };
        });
    }

    //calculation of the body height
    if (
      points &&
      points["left_eye"] &&
      points["right_eye"] &&
      points["right_ankle"] &&
      points["left_ankle"]
    ) {
      let eye_offset = Math.abs(points["left_eye"].x - points["right_eye"].x);
      let height_offset =
        Math.abs(
          points["left_eye"].y +
            points["right_eye"].y -
            (points["left_ankle"].y + points["right_ankle"].y)
        ) / 2;
      height = Math.floor((height_offset / eye_offset) * RATIO);
    }

    //calculation if the body pose is straight or not.
    if (Object.keys(points).length >= 17) {
      isDetected = true;
      isStraight = true;
      console.log("-----------start-------------");
      //eye
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          { x: 0, y: points["left_eye"].y },
          points["left_eye"],
          points["right_eye"]
        );
      //shoulder
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          { x: 0, y: points["left_shoulder"].y },
          points["left_shoulder"],
          points["right_shoulder"]
        );
      //elbow
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          { x: 0, y: points["left_elbow"].y },
          points["left_elbow"],
          points["right_elbow"]
        );
      //wrist
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          { x: 0, y: points["left_wrist"].y },
          points["left_wrist"],
          points["right_wrist"]
        );
      //hip
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          { x: 0, y: points["left_hip"].y },
          points["left_hip"],
          points["right_hip"]
        );
      //knee
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          { x: 0, y: points["left_knee"].y },
          points["left_knee"],
          points["right_knee"]
        );
      //ankle
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          { x: 0, y: points["left_ankle"].y },
          points["left_ankle"],
          points["right_ankle"]
        );

      //hip-knee-ankle
      console.log("-----------hip-knee-ankle-------------");
      // -left
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          points["left_hip"],
          points["left_knee"],
          points["left_ankle"]
        );

      // console.log(points["left_hip"].x, points["left_hip"].y);
      // console.log(points["left_knee"].x, points["left_knee"].y);
      // console.log(points["left_ankle"].x, points["left_ankle"].y);

      // -right
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          points["right_hip"],
          points["right_knee"],
          points["right_ankle"]
        );

      //hip-knee-ankle
      console.log("-----------eye-knee-ankle-------------");
      // -left
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          points["left_eye"],
          points["left_knee"],
          points["left_ankle"]
        );

      // -right
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          points["right_eye"],
          points["right_knee"],
          points["right_ankle"]
        );
      console.log("-----------shoulder-elbow-wrist-------------");
      //shoulder-elbow-wrist
      // -left
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          points["left_shoulder"],
          points["left_elbow"],
          points["left_wrist"]
        );
      // -right
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          points["right_shoulder"],
          points["right_elbow"],
          points["right_wrist"]
        );
      console.log("-----------shoulder-wrist-------------");
      //shoulder-wrist
      // -left
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          { x: points["left_shoulder"].x, y: 0 },
          points["left_shoulder"],
          points["left_wrist"]
        );
      // -right
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          { x: points["right_shoulder"].x, y: 0 },
          points["right_shoulder"],
          points["right_wrist"]
        );

      console.log("-----------hip-ankle-------------");
      //hip-ankle
      // -left
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          { x: points["left_ankle"].x, y: 0 },
          points["left_ankle"],
          points["left_hip"]
        );
      // -right
      isStraight =
        isStraight &&
        arePointsApproximatelyInStraightLine(
          { x: points["right_ankle"].x, y: 0 },
          points["right_ankle"],
          points["right_hip"]
        );
      console.log("-----------end-------------");
    }

    return (
      <View style={styles.heightContainer}>
        <Text>Height: {height}</Text>
        <Text>Straight: {isStraight ? "Yes" : "No"}</Text>
        <Text>Detected: {isDetected ? "Yes" : "No"}</Text>
      </View>
    );
  };

  const renderCameraTypeSwitcher = () => {
    return (
      <View
        style={styles.cameraTypeSwitcher}
        onTouchEnd={handleSwitchCameraType}
      >
        <Text>
          Switch to{" "}
          {cameraType === Camera.Constants.Type.front ? "back" : "front"} camera
        </Text>
      </View>
    );
  };

  const handleSwitchCameraType = () => {
    if (cameraType === Camera.Constants.Type.front) {
      setCameraType(Camera.Constants.Type.back);
    } else {
      setCameraType(Camera.Constants.Type.front);
    }
  };

  const isPortrait = () => {
    return (
      orientation === ScreenOrientation.Orientation.PORTRAIT_UP ||
      orientation === ScreenOrientation.Orientation.PORTRAIT_DOWN
    );
  };

  const getOutputTensorWidth = () => {
    // On iOS landscape mode, switch width and height of the output tensor to
    // get better result. Without this, the image stored in the output tensor
    // would be stretched too much.
    //
    // Same for getOutputTensorHeight below.
    return isPortrait() || IS_ANDROID
      ? OUTPUT_TENSOR_WIDTH
      : OUTPUT_TENSOR_HEIGHT;
  };

  const getOutputTensorHeight = () => {
    return isPortrait() || IS_ANDROID
      ? OUTPUT_TENSOR_HEIGHT
      : OUTPUT_TENSOR_WIDTH;
  };

  const getTextureRotationAngleInDegrees = () => {
    // On Android, the camera texture will rotate behind the scene as the phone
    // changes orientation, so we don't need to rotate it in TensorCamera.
    if (IS_ANDROID) {
      return 0;
    }

    // For iOS, the camera texture won't rotate automatically. Calculate the
    // rotation angles here which will be passed to TensorCamera to rotate it
    // internally.
    switch (orientation) {
      // Not supported on iOS as of 11/2021, but add it here just in case.
      case ScreenOrientation.Orientation.PORTRAIT_DOWN:
        return 180;
      case ScreenOrientation.Orientation.LANDSCAPE_LEFT:
        return cameraType === Camera.Constants.Type.front ? 270 : 90;
      case ScreenOrientation.Orientation.LANDSCAPE_RIGHT:
        return cameraType === Camera.Constants.Type.front ? 90 : 270;
      default:
        return 0;
    }
  };

  if (!tfReady) {
    return (
      <View style={styles.loadingMsg}>
        <Text>Loading...</Text>
      </View>
    );
  } else {
    return (
      // Note that you don't need to specify `cameraTextureWidth` and
      // `cameraTextureHeight` prop in `TensorCamera` below.
      <View
        style={
          isPortrait() ? styles.containerPortrait : styles.containerLandscape
        }
      >
        <TensorCamera
          ref={cameraRef}
          style={styles.camera}
          autorender={AUTO_RENDER}
          type={cameraType}
          // tensor related props
          resizeWidth={getOutputTensorWidth()}
          resizeHeight={getOutputTensorHeight()}
          resizeDepth={3}
          rotation={getTextureRotationAngleInDegrees()}
          onReady={handleCameraStream}
        />
        {renderPose()}
        {renderFps()}
        {renderHeight()}
        {renderCameraTypeSwitcher()}
      </View>
    );
  }
}

const styles = StyleSheet.create({
  containerPortrait: {
    position: "relative",
    width: CAM_PREVIEW_WIDTH,
    height: CAM_PREVIEW_HEIGHT,
    marginTop: Dimensions.get("window").height / 2 - CAM_PREVIEW_HEIGHT / 2,
  },
  containerLandscape: {
    position: "relative",
    width: CAM_PREVIEW_HEIGHT,
    height: CAM_PREVIEW_WIDTH,
    marginLeft: Dimensions.get("window").height / 2 - CAM_PREVIEW_HEIGHT / 2,
  },
  loadingMsg: {
    position: "absolute",
    width: "100%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
  camera: {
    width: "100%",
    height: "100%",
    zIndex: 1,
  },
  svg: {
    width: "100%",
    height: "100%",
    position: "absolute",
    zIndex: 30,
  },
  fpsContainer: {
    position: "absolute",
    top: 10,
    left: 10,
    width: 80,
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, .7)",
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
  heightContainer: {
    position: "absolute",
    top: 60,
    left: 10,
    width: 120,
    alignItems: "flex-start",
    backgroundColor: "rgba(255, 255, 255, .7)",
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
  cameraTypeSwitcher: {
    position: "absolute",
    top: 10,
    right: 10,
    width: 180,
    alignItems: "center",
    backgroundColor: "rgba(255, 255, 255, .7)",
    borderRadius: 2,
    padding: 8,
    zIndex: 20,
  },
});
