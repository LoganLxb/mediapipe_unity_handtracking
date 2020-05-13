/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
using System;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using AOT;
using TfLiteInterpreter = System.IntPtr;
using TfLiteInterpreterOptions = System.IntPtr;
using TfLiteDelegate = System.IntPtr;
using TfLiteModel = System.IntPtr;
using TfLiteTensor = System.IntPtr;
using System.Text;
using UnityEngine;

namespace TensorFlowLite
{
    /// <summary>
    /// Simple C# bindings for the experimental TensorFlowLite C API.
    /// </summary>
    public class Interpreter : IDisposable
    {
#if UNITY_IOS && !UNITY_EDITOR
        private const string TensorFlowLibrary = "__Internal";
        private const string TensorFlowLibraryGPU = "__Internal";
#else
        private const string TensorFlowLibrary = "libtensorflowlite_c";
        private const string TensorFlowLibraryGPU = "libtensorflowlite_gpu_gl";
#endif

        private TfLiteModel model;
        private TfLiteInterpreter interpreter;
        private TfLiteInterpreterOptions interpreterOptions;
        private TfLiteDelegate gpuDelegate;
        private const int numThreads = 2;

        public Interpreter(byte[] modelData, bool useGPU = false) 
        {
            GCHandle modelDataHandle = GCHandle.Alloc(modelData, GCHandleType.Pinned);
            IntPtr modelDataPtr = modelDataHandle.AddrOfPinnedObject();
//            Debug.Log($"TfLite version: {GetVersion()}");
            model = TfLiteModelCreate(modelDataPtr, modelData.Length);
            if (model == IntPtr.Zero){ throw new Exception("Failed to create TensorFlowLite Model"); }
            
            interpreterOptions = TfLiteInterpreterOptionsCreate();
            TfLiteInterpreterOptionsSetNumThreads(interpreterOptions, numThreads);

            TFLInterpreterErrorReporter reporter = InterpreterErrorReporter;
            TfLiteInterpreterOptionsSetErrorReporter(interpreterOptions, reporter, interpreter);

            if(useGPU){ GpuDelegateCreate(); }

            interpreter = TfLiteInterpreterCreate(model, interpreterOptions);
            if (interpreter == IntPtr.Zero)
            { 
                throw new Exception("Failed to create TensorFlowLite Interpreter"); 
            }
            Debug.Log("Interpreter(): Successfully created model and interpreter");
            if(!useGPU){ AllocateTensors(); }
        }

        public void GpuDelegateCreate()
        { 
            var glCompileOptions = new TfLiteGlCompileOptions();
            glCompileOptions.precision_loss_allowed = 0;
            glCompileOptions.preferred_gl_object_type = (Int32)TfLiteGlObjectType.TFLITE_GL_OBJECT_TYPE_FASTEST;
            glCompileOptions.dynamic_batch_enabled = 0;
            glCompileOptions.inline_parameters = 1;

            var gpuDelegateOptions = new TfLiteGpuDelegateOptions();
            gpuDelegateOptions.metadata = IntPtr.Zero;
            gpuDelegateOptions.compile_options = glCompileOptions;

            var metalDelegateOptions = new TfLiteMetalDelegateOptions();
            gpuDelegate = TFLGpuDelegateCreate(metalDelegateOptions);
            if(gpuDelegate == IntPtr.Zero)
            { 
                throw new Exception("TensorFlowLite GPU create failed.");
            }
            TfLiteInterpreterOptionsAddDelegate(interpreterOptions, gpuDelegate);
        }

        public void GpuDelegateDelete(TfLiteDelegate gpuDelegate){ TFLGpuDelegateDelete(gpuDelegate); }


        public void ResizeInputTensor(int inputTensorIndex, int[] inputTensorShape) 
        {
            ThrowIfError(TfLiteInterpreterResizeInputTensor(interpreter, inputTensorIndex, inputTensorShape, inputTensorShape.Length));
        }
        public void AllocateTensors() { ThrowIfError(TfLiteInterpreterAllocateTensors(interpreter)); }

        public int GetInputTensorCount() { return TfLiteInterpreterGetInputTensorCount(interpreter); }

        public TfLiteTensor GetInputTensor(int inputIndex) { return TfLiteInterpreterGetInputTensor(interpreter, inputIndex); }

        public int GetOutputTensorCount() { return TfLiteInterpreterGetOutputTensorCount(interpreter); }

        public TfLiteTensor GetOutputTensor(int outputIndex) { return TfLiteInterpreterGetOutputTensor(interpreter, outputIndex); }

        public TfLiteType GetTensorType(TfLiteTensor tensor) { return TfLiteTensorType(tensor); }

        public int GetTensorNumDims(TfLiteTensor tensor) { return TfLiteTensorNumDims(tensor); }

        public int GetTensorDim(TfLiteTensor tensor, int dimIndex) { return TfLiteTensorDim(tensor, dimIndex); }

        public int GetTensorByteSize(TfLiteTensor tensor) { return TfLiteTensorByteSize(tensor); }

        public IntPtr GetTensorData(TfLiteTensor tensor) { return TfLiteTensorData(tensor); }

        public string GetTensorName(TfLiteTensor tensor) 
        {
            /*
            byte[] str = new byte[0xFF];
            int count = vsprintf(str, TfLiteTensorName(tensor), IntPtr.Zero);
            return Encoding.ASCII.GetString(str, 0, count);
            */
            return Marshal.PtrToStringAnsi(TfLiteTensorName(tensor));
        }

        /// <summary>
        /// Returns a string describing version information of the TensorFlow Lite library.
        /// TensorFlow Lite uses semantic versioning.
        /// </summary>
        /// <returns>A string describing version information</returns>
        public string GetVersion()
        {
            return Marshal.PtrToStringAnsi(TfLiteVersion());
        }


        public TfLiteQuantizationParams GetTensorQuantizationParams(TfLiteTensor tensor) { return TfLiteTensorQuantizationParams(tensor); }

        public void SetInputTensorData(int inputTensorIndex, Array inputTensorData) 
        {
            GCHandle tensorDataHandle = GCHandle.Alloc(inputTensorData, GCHandleType.Pinned);
            IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();
            TfLiteTensor tensor = TfLiteInterpreterGetInputTensor(interpreter, inputTensorIndex);
            ThrowIfError(TfLiteTensorCopyFromBuffer(tensor, tensorDataPtr, Buffer.ByteLength(inputTensorData)));
        }
        public void Invoke(){ ThrowIfError(TfLiteInterpreterInvoke(interpreter)); }

        public void GetOutputTensorData(int outputTensorIndex, Array outputTensorData) 
        {
            GCHandle tensorDataHandle = GCHandle.Alloc(outputTensorData, GCHandleType.Pinned);
            IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();
            TfLiteTensor tensor = TfLiteInterpreterGetOutputTensor(interpreter, outputTensorIndex);
            ThrowIfError(TfLiteTensorCopyToBuffer(tensor, tensorDataPtr, Buffer.ByteLength(outputTensorData)));
        }

        private static unsafe void ThrowIfError(TfLiteStatus resultCode) 
        {
            if (resultCode != TfLiteStatus.kTfLiteOk)
            {
                throw new Exception("TensorFlowLite operation failed.");
            }
        }

        ~Interpreter() { Dispose(); }

        public void Dispose() 
        {
//wip
//            if(gpuDelegate != null){ TfLiteGpuDelegateDelete(gpuDelegate); }

            if (interpreterOptions != IntPtr.Zero){ TfLiteInterpreterOptionsDelete(interpreterOptions); }
            interpreterOptions = IntPtr.Zero;

            if (interpreter != IntPtr.Zero){ TfLiteInterpreterDelete(interpreter); }
            interpreter = IntPtr.Zero;

            if (model != IntPtr.Zero){ TfLiteModelDelete(model); }
            model = IntPtr.Zero;
        }

#region Externs
        public enum TfLiteStatus { kTfLiteOk = 0, kTfLiteError = 1 };
        public enum TfLiteType 
        {
            kTfLiteNoType = 0, kTfLiteFloat32 = 1, kTfLiteInt32 = 2, kTfLiteUInt8 = 3, kTfLiteInt64 = 4,
            kTfLiteString = 5, kTfLiteBool = 6, kTfLiteInt16 = 7, kTfLiteComplex64 = 8, kTfLiteInt8 = 9, kTfLiteFloat16 = 10,
        };
        public struct TfLiteQuantizationParams 
        {
            public float scale;
            public Int32 zero_point;
        };
        
        public enum TfLiteGlObjectType
        {
            TFLITE_GL_OBJECT_TYPE_FASTEST = 0,
            TFLITE_GL_OBJECT_TYPE_TEXTURE = 1,
            TFLITE_GL_OBJECT_TYPE_BUFFER = 2,
        };
        public struct TfLiteGlCompileOptions 
        {
            public Int32 precision_loss_allowed;
            public Int32 preferred_gl_object_type;
            public Int32 dynamic_batch_enabled;
            public Int32 inline_parameters;
        };
        public unsafe struct TfLiteGpuDelegateOptions 
        {
            public IntPtr metadata;
            public TfLiteGlCompileOptions compile_options;
        };

        public enum WaitType
        {
            Passive = 0,
            Active = 1,
            DoNotWait = 2,
            Aggressive = 3,
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct TfLiteMetalDelegateOptions
        {
            public bool allowPrecisionLoss;
            public WaitType waitType;
        }

        public enum TfLiteDelegateFlags 
        {
            kTfLiteDelegateFlagsNone = 0,
            kTfLiteDelegateFlagsAllowDynamicTensors = 1
        };

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void TFLInterpreterErrorReporter(IntPtr userData, string format, IntPtr args);
        [MonoPInvokeCallback(typeof(TFLInterpreterErrorReporter))]
        private static void InterpreterErrorReporter(IntPtr userData, string format, IntPtr args)
        {
//            byte[] str = new byte[0xFF];
//            int count = vsprintf(str, format, args);
//            Debug.LogError(Encoding.ASCII.GetString(str, 0, count));
                Debug.Log($"Error: {format}");
        }

        [DllImport("msvcrt")]
        private static extern unsafe int vsprintf(byte[] str, string format, IntPtr args);

        [DllImport("msvcrt")]
        private static extern unsafe int vsprintf(byte[] str, IntPtr format, IntPtr args);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreter TfLiteModelCreate(IntPtr modelData, Int32 modelSize);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreter TfLiteModelDelete(TfLiteModel model);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreterOptions TfLiteInterpreterOptionsCreate();

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions options);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions options, Int32 numThreads);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsAddDelegate(TfLiteInterpreterOptions options, TfLiteDelegate delegate_);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterOptionsSetErrorReporter(TfLiteInterpreterOptions options, 
                                                                                    TFLInterpreterErrorReporter reporter, IntPtr userData);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteInterpreter TfLiteInterpreterCreate(TfLiteModel model, TfLiteInterpreterOptions optionalOptions);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe void TfLiteInterpreterDelete(TfLiteInterpreter interpreter);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe Int32 TfLiteInterpreterGetInputTensorCount(TfLiteInterpreter interpreter);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteTensor TfLiteInterpreterGetInputTensor(TfLiteInterpreter interpreter, Int32 inputIndex);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteStatus TfLiteInterpreterResizeInputTensor(TfLiteInterpreter interpreter, Int32 inputIndex, 
                                                                                    Int32[] inputDims, Int32 inputDimsSize);
        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteStatus TfLiteInterpreterAllocateTensors(TfLiteInterpreter interpreter);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteStatus TfLiteInterpreterInvoke(TfLiteInterpreter interpreter);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe Int32 TfLiteInterpreterGetOutputTensorCount(TfLiteInterpreter interpreter);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteTensor TfLiteInterpreterGetOutputTensor(TfLiteInterpreter interpreter, Int32 outputIndex);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteType TfLiteTensorType(TfLiteTensor tensor);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe Int32 TfLiteTensorNumDims(TfLiteTensor tensor);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe Int32 TfLiteTensorDim(TfLiteTensor tensor, Int32 dimIndex);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe Int32 TfLiteTensorByteSize(TfLiteTensor tensor);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe IntPtr TfLiteTensorData(TfLiteTensor tensor);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe IntPtr TfLiteTensorName(TfLiteTensor tensor);
        [DllImport (TensorFlowLibrary)]
        private static extern unsafe IntPtr TfLiteVersion();

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteQuantizationParams TfLiteTensorQuantizationParams(TfLiteTensor tensor);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteStatus TfLiteTensorCopyFromBuffer(TfLiteTensor tensor, IntPtr inputData, Int32 inputDataSize);

        [DllImport (TensorFlowLibrary)]
        private static extern unsafe TfLiteStatus TfLiteTensorCopyToBuffer(TfLiteTensor tensor, IntPtr outputData, Int32 outputDataSize);

        [DllImport (TensorFlowLibraryGPU)]
        private static extern unsafe TfLiteDelegate TFLGpuDelegateCreate(TfLiteMetalDelegateOptions delegateOptions);

        [DllImport (TensorFlowLibraryGPU)]
        private static extern unsafe void TFLGpuDelegateDelete(TfLiteDelegate delegate_);

#endregion
    }
}