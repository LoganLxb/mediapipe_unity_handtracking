# mediapipe_unity_handtracking
# TF Lite Experimental Unity Plugin

This directory contains an experimental sample Unity Plugin, based on
the experimental TF Lite C API. The sample demonstrates running inference within
Unity by way of a C# `Interpreter` wrapper.

## Changes to the original repo

Used Plugins from asus4
https://github.com/asus4/tf-lite-unity-sample/tree/master/Assets/TensorFlowLite/Plugins

Interpreter script modified to use new plugins and added iOS compilation flags required for DllImport().
Also handled TFLInterpreterErrorReporter() callback properly using PInvokeCallback.
Removed crash due to vsprintf() in both InterpreterErrorReporter() and GetTensorName().
Added a function GetVersion() to obtain the Tensorflow Lite version used.

HandTracking and DebugRenderer scripts changed to cater iOS camera image rotation.

## Verified platform

Unity 2019.3.9, iOS 13.4.1 verified

## To run the demo

* Clone or download this repo
* Open from Unity (choose a verion at or above 2019.3.3)
* Open the HandTracking Scene in Asset/TensorFlowLite/Examples/HandTracking/Scenes
* Open Build Settings... from File menu in Unity
* Choose iOS and click Switch Ploatform
* Uncheck all from Scenes in Build and click Add Open Scenes
* Click Player Settings...
* In Player Settings, make sure the followings:
  * Auto Graphics API is Not checked
  * API compatibility Level is .NET 4.x
  * Target minimum iOS version is 13.4
  * Architecture is ARM64
  * Allow 'unsafe' code is Checked
  * Strip Engine Code is Not checked
* Go back to Build Settings and click Build and Run


## How to build tensorflow lite for Unity

Pre-build library is included. see following instructions if you want to build your own lib.

### macOS

```sh
# Core Lib
bazel build -c opt --cxxopt=--std=c++11 tensorflow/lite/experimental/c:libtensorflowlite_c.so

# Use this branch to build metal GPU delegate dynamic library
# https://github.com/asus4/tensorflow/tree/tflite-macos-metal-delegate
bazel 'build' -c opt --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always --cxxopt=-std=c++14 --apple_platform_type=macos '//tensorflow/lite/delegates/gpu:tensorflow_lite_gpu_dylib'
```

then rename libtensorflowlite_c.so to libtensorflowlite_c.bundle

### iOS

Download pre-build framework from CocoaPods

```ruby
# Sample Podfile

platform :ios, '10.0'

target 'TfLiteSample' do
    pod 'TensorFlowLiteObjC', '0.0.1-nightly'
end
```

```sh
# and build Metal GPU delegete with bitcode option enabled
bazel build -c opt --cpu ios_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --copt=-fembed-bitcode --linkopt -s --strip always --cxxopt=-std=c++14 //tensorflow/lite/delegates/gpu:tensorflow_lite_gpu_framework --apple_platform_type=ios
```

### Android

If you do not have the Android SDK and NDK, intall Android Studio, SDK and NDK.

```sh
# Configure the Android SDK path by running configure script at repository root
./configure

# Build experimental
bazel build -c opt --cxxopt=--std=c++11 --config=android_arm64 //tensorflow/lite/experimental/c:libtensorflowlite_c.so

# Build GPU delegate
bazel build -c opt --config android_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always //tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so
```
