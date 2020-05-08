
using System;
using TensorFlowLite;
using UnityEngine;

using static UnityEngine.Mathf;

// Ref: https://github.com/google/mediapipe/blob/master/mediapipe/graphs/hand_tracking/subgraphs/hand_detection_gpu.pbtxt
// Ref: https://github.com/google/mediapipe/blob/master/mediapipe/graphs/hand_tracking/subgraphs/hand_landmark_gpu.pbtxt
public class Inferencer 
{
    private const int NN_INPUT_WIDTH = 256;
    private const int NN_INPUT_HEIGHT = 256;
    private const int NN_INPUT_CHANNEL = 3;
    private const int NN_NUM_CLASSES = 1;
    private const int NN_NUM_BOXES = 2944;
    private const int NN_NUM_BOX_SIZE = 4;
    private const int NN_NUM_COORDS = 18; // box(x,y,z,w) + keypoints(x,y) * 7
    private const int NN_NUM_KEYPOINTS = 7;
    private const int NN_NUM_VALUES_PERKEYPOINT = 2;
    private const int NN_NUM_KEYPOINTS_SIZE = NN_NUM_KEYPOINTS * NN_NUM_VALUES_PERKEYPOINT;

    private const int NN_NUM_LANDMARKS = 21;

    private Interpreter palmDetectionInterpreter;
    private Interpreter handLandmarksInterpreter;

    private float[,,,] palmDetectionInputs = new float[1, NN_INPUT_HEIGHT, NN_INPUT_WIDTH, NN_INPUT_CHANNEL];
    private float[,,,] handLandmarksInputs = new float[1, NN_INPUT_HEIGHT, NN_INPUT_WIDTH, NN_INPUT_CHANNEL];
    
    private float[] normalizedRGB = new float[256];
    private float[,] anchorsOutputs = new float[NN_NUM_BOXES, NN_NUM_BOX_SIZE];
    private float[,,] regressorsOutputs = new float[1, NN_NUM_BOXES, NN_NUM_COORDS];
    private float[,,] classificatorsOutputs = new float[1, NN_NUM_BOXES, NN_NUM_CLASSES];
    private float[,] landmarksOutputs = new float[1, NN_NUM_LANDMARKS * 3];
    private float[,] handFlagOutputs = new float[1, 1];

    public bool Initialized = false;
    public int InputWidth { get { return NN_INPUT_WIDTH; } }
    public int InputHeight { get { return NN_INPUT_HEIGHT; } }
    public float[,,,] PalmDetectionInputs { get { return palmDetectionInputs; } }
    public int PalmNumKeypoints { get { return NN_NUM_KEYPOINTS; } }
    public float[,,,] HandLandmarksInputs { get { return handLandmarksInputs; } }

    public Rect PalmBox = new Rect();
    public Vector2[] PalmKeypoints = new Vector2[NN_NUM_KEYPOINTS];
    public Vector3[] HandLandmarks = new Vector3[NN_NUM_LANDMARKS];

    public void Init(TextAsset palmDetection, TextAsset handLandmarks, bool useGPU,
                        int palmDetectionLerpFrameCount, int handLandmark3DLerpFrameCount) 
    {
        palmDetectionInterpreter = InitInterpreter(palmDetection, useGPU);
        handLandmarksInterpreter = InitInterpreter(handLandmarks, useGPU);
        InitAnchors(anchorsOutputs, NN_INPUT_WIDTH, NN_INPUT_HEIGHT);

        // Image inputs are normalized to [-1,1]
        const float ItoF = 1.0f / 255.0f;
        for(int i = 0; i< 256; ++i){ normalizedRGB[i] = (i * ItoF * 2.0f) - 1.0f; }

        lerpPalmRectFrameCount = palmDetectionLerpFrameCount;
        lerpHandLandmarkFrameCount = handLandmark3DLerpFrameCount;
    }
    private Interpreter InitInterpreter(TextAsset model, bool useGPU)
    { 
        var interpreter = new Interpreter(model.bytes, useGPU);

        var inputTensorCount = interpreter.GetInputTensorCount();
        for(int i = 0; i < inputTensorCount; ++i){ DebugTensorData(interpreter, interpreter.GetInputTensor(i)); }

        var outputTensorCount = interpreter.GetOutputTensorCount();
        for(int i = 0; i < outputTensorCount; ++i){ DebugTensorData(interpreter, interpreter.GetOutputTensor(i)); }

        return interpreter;
    }
    private void DebugTensorData(Interpreter interpreter, IntPtr tensor) 
    { 
        var type = interpreter.GetTensorType(tensor);
        int numDims = interpreter.GetTensorNumDims(tensor);
        int[] dims = new int[numDims];
        for(int i = 0; i < numDims; ++i) { dims[i] = interpreter.GetTensorDim(tensor, i); }
        int byteSize = interpreter.GetTensorByteSize(tensor);
        IntPtr data = interpreter.GetTensorData(tensor);
        var name = interpreter.GetTensorName(tensor);
        var tensorQuantizationParams = interpreter.GetTensorQuantizationParams(tensor);
    }

    private void InitAnchors(float[,] anchors, int width, int height)
    {
        const int SSD_NUM_LAYERS = 5;
        const float SSD_ANCHOR_OFFSET_X = 0.5f;
        const float SSD_ANCHOR_OFFSET_Y = 0.5f;
        int[] SSD_STRIDES = { 8, 16, 32, 32, 32 };

        int index = 0;
        for(int layer = 0; layer < SSD_NUM_LAYERS; ++layer)
        {
            float stride = SSD_STRIDES[layer];
            int featureMapHeight = CeilToInt(width / stride);
            int featureMapWidth = CeilToInt(height / stride);
            for(int y = 0; y < featureMapHeight; ++y)
            {
                float centerY = (y + SSD_ANCHOR_OFFSET_Y) / featureMapHeight;
                for(int x = 0; x < featureMapWidth; ++x)
                {
                    float centerX = (x + SSD_ANCHOR_OFFSET_X) / featureMapWidth;
                    for(int anchorID = 0; anchorID < 2; ++anchorID)
                    { 
                        anchors[index, 0] = centerX;
                        anchors[index, 1] = centerY;
                        anchors[index, 2] = 1.0f;
                        anchors[index, 3] = 1.0f;
                        ++index;
                    }
                }
            }
        }
    }

    public void Update(Texture2D texture) 
    {
        var size = new Vector2(texture.width, texture.height);
        var center = new Vector2(size.x * 0.5f, texture.height * 0.5f);
        var angle = 0.0f * Deg2Rad;

        CreateShapes(texture, palmDetectionInputs, size, center, angle);
        DetectePalmRect();
        CalcPalmRect();
        CalcHandRect();
        LerpPalmRect();

        CreateShapes(texture, handLandmarksInputs, HandSize, HandCenter, handAngle);
        DetecteLandmarksPos();
        CalcLandmarksPos();
        LerpLandmarksPos();

        Initialized = true;
    }

    private const bool shapeFlipX = true;
    private const bool shapeFlipY = true;
    private unsafe void CreateShapes(Texture2D texture, float[,,,] inputs,
                                        Vector2 size, Vector2 center, float angle) 
    {
        const int RGB = 3; // TextureFormat.RGB24
        byte[] pixels = texture.GetRawTextureData();

        int srcW = texture.width, srcH = texture.height; 
        int dstW = NN_INPUT_WIDTH, dstH = NN_INPUT_HEIGHT; 
        float cropW = size.x, cropH = size.y;
        float scaleW = (float)dstW / srcW, scaleH = (float)dstH / srcH; 

        float longSide = (srcW >= srcH) ? srcW : srcH; 
        float padScaleW = srcW / longSide, padScaleH = srcH / longSide; 
        int dstPadW = (int)((longSide - srcW) * padScaleW * scaleW * 0.5f); 
        int dstPadH = (int)((longSide - srcH) * padScaleH * scaleH * 0.5f); 
        float invDstH = 1.0f / (dstH - dstPadH * 2.0f);
        float invDstW = 1.0f / (dstW - dstPadW * 2.0f);

        float srcHalfX = (cropW * 0.5f), srcHalfY = (cropH * 0.5f);
        float c = Cos(angle), s = Sin(angle);

        Array.Clear(inputs, 0, inputs.Length);
        fixed (byte* src = pixels) 
        {
            fixed (float* dst = inputs) 
            {
                for (int dstY = dstPadH; dstY < dstH - dstPadH; ++dstY) 
                {
                    float dstV = (dstY - dstPadH) * invDstH;
                    float srcLocalY = (cropH * dstV) - srcHalfY;

                    float* dstPos = dst + (dstY * dstW + dstPadW) * NN_INPUT_CHANNEL;
                    for (int dstX = dstPadW; dstX < dstW - dstPadW; ++dstX) 
                    {
                        float dstU = (dstX - dstPadW) * invDstW;
                        float srcLocalX = (cropW * dstU) - srcHalfX;
                        int srcGlobalX = (int)(center.x + (srcLocalX * c - srcLocalY * s));
                        int srcGlobalY = (int)(center.y + (srcLocalX * s + srcLocalY * c));

                        int srcX = shapeFlipX ? (srcW - 1) - srcGlobalX : srcGlobalX;
                        int srcY = shapeFlipY ? (srcH - 1) - srcGlobalY : srcGlobalY;
                        if(srcX < 0 || srcX >= srcW) { continue; }
                        if(srcY < 0 || srcY >= srcH) { continue; }

                        byte* srcPos = src + (srcY * srcW + srcX) * RGB;
                        *(dstPos++) = normalizedRGB[*(srcPos++)];
                        *(dstPos++) = normalizedRGB[*(srcPos++)];
                        *(dstPos++) = normalizedRGB[*(srcPos++)];
                    }
                }
            }
        }
    }

    private float[] palmScoreCandidates = new float[NN_NUM_BOXES];
    private float[,] palmBoxCandidates = new float[NN_NUM_BOXES, NN_NUM_BOX_SIZE];
    private float[,,] palmKeypointsCandidates = new float[NN_NUM_BOXES, NN_NUM_KEYPOINTS, NN_NUM_VALUES_PERKEYPOINT];
    private float palmBoxMaxScoreX, palmBoxMaxScoreY, palmBoxMaxScoreW, palmBoxMaxScoreH;

    private void DetectePalmRect() 
    { 
        const int NN_BOX_COORD_OFFSET = 0;
        const int NN_KEYPOINT_COORD_OFFSET = 4;
        const float NN_SCORE_CLIPPING_THRESH = 100.0f;
        const float NN_MIN_SCORE_THRESH = 0.7f;
        const float NN_X_SCALE = 256.0f, NN_Y_SCALE = 256.0f;
        const float NN_H_SCALE = 256.0f, NN_W_SCALE = 256.0f;
        const int X = 0, Y = 1, W = 2, H = 3;

        palmDetectionInterpreter.SetInputTensorData(0, palmDetectionInputs);

        float startTimeSeconds = Time.realtimeSinceStartup;
        palmDetectionInterpreter.Invoke();
        float inferenceTimeSeconds = (Time.realtimeSinceStartup - startTimeSeconds) * 10000;
        //Debug.Log(string.Format("Palm detection {0:0.0000} ms", inferenceTimeSeconds));

        Array.Clear(regressorsOutputs, 0, regressorsOutputs.Length);
        palmDetectionInterpreter.GetOutputTensorData(0, regressorsOutputs);
        Array.Clear(classificatorsOutputs, 0, classificatorsOutputs.Length);
        palmDetectionInterpreter.GetOutputTensorData(1, classificatorsOutputs);

        Array.Clear(palmScoreCandidates, 0, palmScoreCandidates.Length);
        Array.Clear(palmBoxCandidates, 0, palmBoxCandidates.Length);
        Array.Clear(palmKeypointsCandidates, 0, palmKeypointsCandidates.Length);

        float maxScore = -1.0f;
        for(int i = 0; i < NN_NUM_BOXES; ++i)
        {
            float score = classificatorsOutputs[0, i, 0];
            score = Clamp(score, -NN_SCORE_CLIPPING_THRESH, NN_SCORE_CLIPPING_THRESH);
            score = Sigmoid(score);

            if(score < NN_MIN_SCORE_THRESH){ continue; }
            palmScoreCandidates[i] = score;

            float centerX = regressorsOutputs[0, i, NN_BOX_COORD_OFFSET + 0];
            float centerY = regressorsOutputs[0, i, NN_BOX_COORD_OFFSET + 1];
            float w = regressorsOutputs[0, i, NN_BOX_COORD_OFFSET + 2];
            float h = regressorsOutputs[0, i, NN_BOX_COORD_OFFSET + 3];

            // anchors[i, 2] and anchors[i, 3] are always 1.0f
            centerX = centerX / NN_X_SCALE * anchorsOutputs[i, 2] + anchorsOutputs[i, 0];
            centerY = centerY / NN_Y_SCALE * anchorsOutputs[i, 3] + anchorsOutputs[i, 1];
            w = w / NN_W_SCALE * anchorsOutputs[i, 2];
            h = h / NN_H_SCALE * anchorsOutputs[i, 3];

            float boxMinY = centerY - h * 0.5f, boxMinX = centerX - w * 0.5f;
            float boxMaxY = centerY + h * 0.5f, boxMaxX = centerX + w * 0.5f;
            palmBoxCandidates[i, X] = boxMinX;
            palmBoxCandidates[i, Y] = boxMinY;
            palmBoxCandidates[i, W] = boxMaxX - boxMinX;
            palmBoxCandidates[i, H] = boxMaxY - boxMinY;

            for(int j = 0; j < NN_NUM_KEYPOINTS; ++j)
            { 
                int ofset = NN_BOX_COORD_OFFSET + NN_KEYPOINT_COORD_OFFSET + j * NN_NUM_VALUES_PERKEYPOINT;
                float keypointX = regressorsOutputs[0, i, ofset + 0];
                float keypointY = regressorsOutputs[0, i, ofset + 1];
                keypointX = keypointX / NN_X_SCALE * anchorsOutputs[i, 2] + anchorsOutputs[i, 0];
                keypointY = keypointY / NN_Y_SCALE * anchorsOutputs[i, 3] + anchorsOutputs[i, 1];
                palmKeypointsCandidates[i, j, 0] = keypointX;
                palmKeypointsCandidates[i, j, 1] = keypointY;
            }

            if(score < maxScore){ continue; }
            maxScore = score;
            palmBoxMaxScoreX = palmBoxCandidates[i, X];
            palmBoxMaxScoreY = palmBoxCandidates[i, Y];
            palmBoxMaxScoreW = palmBoxCandidates[i, W];
            palmBoxMaxScoreH = palmBoxCandidates[i, H];
        }
    }
    private float Sigmoid(float x){ return 1.0f / (1.0f + Exp(-x)); }

    private float[,] totalKeypoints = new float[NN_NUM_KEYPOINTS, NN_NUM_VALUES_PERKEYPOINT];
    private void CalcPalmRect() 
    { 
        const float MIN_SUPPRESSION_THRESHOLD = 0.3f;
        const int X = 0, Y = 1, W = 2, H = 3;

        float totalX = 0.0f, totalY = 0.0f, totalW = 0.0f, totalH = 0.0f;
        float totalScore = 0.0f;
        Array.Clear(totalKeypoints, 0, totalKeypoints.Length);

        for(int i = 0; i < NN_NUM_BOXES; ++i)
        {
            float score = palmScoreCandidates[i];
            float x = palmBoxCandidates[i, X];
            float y = palmBoxCandidates[i, Y];
            float w = palmBoxCandidates[i, W];
            float h = palmBoxCandidates[i, H];
            float similarity = OverlapSimilarity(x, y, w, h);
            if (similarity < MIN_SUPPRESSION_THRESHOLD) { continue; }

            totalScore += score;
            totalX += x * score;
            totalY += y * score;
            totalW += w * score;
            totalH += h * score;
            for(int j = 0; j < NN_NUM_KEYPOINTS; ++j)
            {
                totalKeypoints[j, 0] += palmKeypointsCandidates[i, j, 0] * score;
                totalKeypoints[j, 1] += palmKeypointsCandidates[i, j, 1] * score;
            }
        }

        if(totalScore == 0.0f) { return; }
        float invTotalScore = 1.0f / totalScore;
        PalmBox.Set(totalX * invTotalScore, totalY * invTotalScore, 
                    totalW * invTotalScore, totalH * invTotalScore);

        for(int i = 0; i < NN_NUM_KEYPOINTS; ++i)
        {
            PalmKeypoints[i].Set(totalKeypoints[i, 0] * invTotalScore, 
                                    totalKeypoints[i, 1] * invTotalScore);
        }
    }
    private float OverlapSimilarity(float x, float y, float w, float h) 
    {
        float minX = Max(palmBoxMaxScoreX, x), maxX = Min(palmBoxMaxScoreX + palmBoxMaxScoreW, x + w);
        float minY = Max(palmBoxMaxScoreY, y), maxY = Min(palmBoxMaxScoreY + palmBoxMaxScoreH, y + h);
        if (minX > maxX || minY > maxY) { return 0.0f; }

        float aArea = palmBoxMaxScoreW * palmBoxMaxScoreH, bArea = w * h;
        float intersectionArea = (maxX - minX) * (maxY - minY);
        float normalization = aArea + bArea - intersectionArea;
        if(normalization == 0.0f) { return 0.0f; }

        return intersectionArea / normalization;
    }
 
    private Rect[] lerpPalmBox = null;
    private Vector2[,] lerpPalmKeypoints = null;
    private int lerpPalmRectFrameCount = 3;
    private int lerpPalmRectFrameNum = 0;
    private void LerpPalmRect()
    {
        if (lerpPalmBox == null) 
        {
            lerpPalmBox = new Rect[lerpPalmRectFrameCount];
            lerpPalmKeypoints = new Vector2[lerpPalmRectFrameCount, NN_NUM_KEYPOINTS];
            for(int i = 0; i < lerpPalmRectFrameCount; ++i)
            { 
                lerpPalmBox[i] = PalmBox;
                for(int j = 0; j < NN_NUM_KEYPOINTS; ++j) { lerpPalmKeypoints[i, j] = PalmKeypoints[j]; }
            }

        } else {
            int i = lerpPalmRectFrameNum;
            lerpPalmBox[i] = PalmBox;
            for(int j = 0; j < NN_NUM_KEYPOINTS; ++j) { lerpPalmKeypoints[i, j] = PalmKeypoints[j]; }
        }

        lerpPalmRectFrameNum = (lerpPalmRectFrameNum + 1) % lerpPalmRectFrameCount;

        PalmBox.Set(0, 0, 0, 0);
        for(int j = 0; j < NN_NUM_KEYPOINTS; ++j) { PalmKeypoints[j].Set(0, 0); }
        for(int i = 0; i < lerpPalmRectFrameCount; ++i)
        { 
            PalmBox.x += lerpPalmBox[i].x;
            PalmBox.y += lerpPalmBox[i].y;
            PalmBox.width += lerpPalmBox[i].width;
            PalmBox.height += lerpPalmBox[i].height;
            for(int j = 0; j < NN_NUM_KEYPOINTS; ++j) {  PalmKeypoints[j] += lerpPalmKeypoints[i, j]; }
        }
        PalmBox.x /= lerpPalmRectFrameCount;
        PalmBox.y /= lerpPalmRectFrameCount;
        PalmBox.width /= lerpPalmRectFrameCount;
        PalmBox.height /= lerpPalmRectFrameCount;
        for(int j = 0; j < NN_NUM_KEYPOINTS; ++j) { PalmKeypoints[j] /= lerpPalmRectFrameCount; }
    }

    public Vector2[] HandBox = new Vector2[NN_NUM_BOX_SIZE];
    public Vector2 HandSize, HandCenter;
    private float handAngle = 0.0f;
    private float handCos = 0.0f, handSin = 0.0f;
    private void CalcHandRect() 
    { 
        const int START_KEYPOINT = 0; // Center of wrist.
        const int END_KEYPOINT = 2; // MCP of middle finger.
        const float ANGLE = PI * 90.0f / 180.0f;
        const float SCALE_X = 2.6f, SCALE_Y = 2.6f;
        const float SHIFT_X = 0.0f, SHIFT_Y = -0.5f;

        float startX = PalmKeypoints[START_KEYPOINT].x * NN_INPUT_WIDTH;
        float startY = PalmKeypoints[START_KEYPOINT].y * NN_INPUT_HEIGHT;
        float endX = PalmKeypoints[END_KEYPOINT].x * NN_INPUT_WIDTH;
        float endY = PalmKeypoints[END_KEYPOINT].y * NN_INPUT_HEIGHT;
        float angle = ANGLE - Atan2(-(endY - startY), endX - startX);
        handAngle = angle - 2.0f * PI * Floor((angle - (-PI)) / (2.0f * PI));
        handCos = Cos(handAngle);
        handSin = Sin(handAngle);

        float w = NN_INPUT_WIDTH * PalmBox.width;
        float h = NN_INPUT_HEIGHT * PalmBox.height;

        float centerX = (PalmBox.x + PalmBox.width * 0.5f) * NN_INPUT_WIDTH;
        float centerY = (PalmBox.y + PalmBox.height * 0.5f) * NN_INPUT_HEIGHT;
        if (handAngle == 0.0f) 
        {
            centerX += w * SHIFT_X;
            centerY += h * SHIFT_Y;

        } else { 
            centerX += (w * SHIFT_X * handCos - h * SHIFT_Y * handSin);
            centerY += (w * SHIFT_X * handSin + h * SHIFT_Y * handCos);
        }

        float longSide = Max(w, h);
        float width = (longSide / NN_INPUT_WIDTH) * SCALE_X * NN_INPUT_WIDTH;;
        float height = (longSide / NN_INPUT_HEIGHT) * SCALE_Y * NN_INPUT_HEIGHT;
        float cw = handCos * width * 0.5f, ch = handCos * height * 0.5f;
        float sw = handSin * width * 0.5f, sh = handSin * height * 0.5f;

        HandSize.Set(width, height);
        HandCenter.Set(centerX, centerY);
        HandBox[0].Set(centerX + (-cw - -sh), centerY + (-sw + -ch));
        HandBox[1].Set(centerX + (+cw - -sh), centerY + (+sw + -ch));
        HandBox[2].Set(centerX + (+cw - +sh), centerY + (+sw + +ch));
        HandBox[3].Set(centerX + (-cw - +sh), centerY + (-sw + +ch));
    }

    private Vector3[] landmarks = new Vector3[NN_NUM_LANDMARKS];
    private void DetecteLandmarksPos() 
    { 
        const float NN_THRESHOLD = 0.1f;

        handLandmarksInterpreter.SetInputTensorData(0, handLandmarksInputs);

        float startTimeSeconds = Time.realtimeSinceStartup;
        handLandmarksInterpreter.Invoke();
        float inferenceTimeSeconds = (Time.realtimeSinceStartup - startTimeSeconds) * 10000;
        //Debug.Log(string.Format("Palm detection {0:0.0000} ms", inferenceTimeSeconds));

        Array.Clear(handFlagOutputs, 0, handFlagOutputs.Length);
        handLandmarksInterpreter.GetOutputTensorData(1, handFlagOutputs);

        if(handFlagOutputs[0, 0] < NN_THRESHOLD){ return; }

        Array.Clear(landmarksOutputs, 0, landmarksOutputs.Length);
        handLandmarksInterpreter.GetOutputTensorData(0, landmarksOutputs);

        float invW = 1.0f / NN_INPUT_WIDTH, invH = 1.0f / NN_INPUT_HEIGHT;
        for(int i = 0; i < NN_NUM_LANDMARKS; ++i)
        { 
            float x = landmarksOutputs[0, i * 3] * invW;
            float y = landmarksOutputs[0, i * 3 + 1] * invH;
            float z = landmarksOutputs[0, i * 3 + 2];
            landmarks[i].Set(x, y, z);
         }
    }
    private void CalcLandmarksPos()
    {
        float w = HandSize.x, h = HandSize.y;
        for(int i = 0; i < NN_NUM_LANDMARKS; ++i)
        {
            float halfX = w * (landmarks[i].x - 0.5f);
            float halfY = h * (landmarks[i].y - 0.5f);
            float x = HandCenter.x + (halfX * handCos - halfY * handSin);
            float y = HandCenter.y + (halfX * handSin + halfY * handCos);
            float z = landmarks[i].z;
            HandLandmarks[i].Set(x, y, z);
        }
    }

    private Vector3[,] lerpHandLandmarks = null;
    private int lerpHandLandmarkFrameCount = 4;
    private int lerpHandLandmarkFrameNum = 0;
    private void LerpLandmarksPos()
    {
        if (lerpHandLandmarks == null) 
        {
            lerpHandLandmarks = new Vector3[lerpHandLandmarkFrameCount, NN_NUM_LANDMARKS];
            for(int i = 0; i < lerpHandLandmarkFrameCount; ++i)
            { 
                for(int j = 0; j < NN_NUM_LANDMARKS; ++j) { lerpHandLandmarks[i, j] = HandLandmarks[j]; }
            }

        } else {
            int i = lerpHandLandmarkFrameNum;
            for(int j = 0; j < NN_NUM_LANDMARKS; ++j) { lerpHandLandmarks[i, j] = HandLandmarks[j]; }
        }

        lerpHandLandmarkFrameNum = (lerpHandLandmarkFrameNum + 1) % lerpHandLandmarkFrameCount;

        for(int j = 0; j < NN_NUM_LANDMARKS; ++j) { HandLandmarks[j].Set(0, 0, 0); }
        for(int i = 0; i < lerpHandLandmarkFrameCount; ++i)
        { 
            for(int j = 0; j < NN_NUM_LANDMARKS; ++j) {  HandLandmarks[j] += lerpHandLandmarks[i, j]; }
        }
        for(int j = 0; j < NN_NUM_LANDMARKS; ++j) {  HandLandmarks[j] /= lerpHandLandmarkFrameCount; }
    }

    public void Destroy() 
    { 
        if(palmDetectionInterpreter != null){ palmDetectionInterpreter.Dispose(); }
        if(handLandmarksInterpreter != null){ handLandmarksInterpreter.Dispose(); }
    }
}
