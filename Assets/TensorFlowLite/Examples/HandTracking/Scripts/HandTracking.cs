
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;

public class HandTracking : MonoBehaviour 
{
    [Tooltip("Configurable TFLite model.")]
    public int InputW = 256;
    public int InputH = 256;
    public TextAsset PalmDetection;
    public TextAsset HandLandmark3D;
    public int PalmDetectionLerpFrameCount = 3;
    public int HandLandmark3DLerpFrameCount = 4;
    public bool UseGPU = true;
    private RenderTexture videoTexture;
    private Texture2D texture;

    private Inferencer inferencer = new Inferencer();
    private GameObject debugPlane;
    private DebugRenderer debugRenderer;


    private string deviceName;
    private WebCamTexture webCam;

    void Awake() { QualitySettings.vSyncCount = 0; }

    void Start() 
    {
        OpenCamera();
        //InitTexture();
        inferencer.Init(PalmDetection, HandLandmark3D, UseGPU,
                        PalmDetectionLerpFrameCount, HandLandmark3DLerpFrameCount);
        debugPlane = GameObject.Find("TensorFlowLite");
        debugRenderer = debugPlane.GetComponent<DebugRenderer>();
        debugRenderer.Init(inferencer.InputWidth, inferencer.InputHeight, debugPlane);
    }
    private void InitTexture()
    { 
        var rectTransform = GetComponent<RectTransform>();
        var renderer = GetComponent<Renderer>();

        var videoPlayer = GetComponent<VideoPlayer>();
        int width = (int)rectTransform.rect.width;
        int height = (int)rectTransform.rect.height;
        videoTexture = new RenderTexture(width, height, 24);
        videoPlayer.targetTexture = videoTexture;
        renderer.material.mainTexture = videoTexture;
        videoPlayer.Play();

        texture = new Texture2D(videoTexture.width, videoTexture.height, TextureFormat.RGB24, false);
    }

    void OpenCamera()
    {
        WebCamDevice [] devices= WebCamTexture.devices;
        if (devices.Length>0)
        {

            var rectTransform = GetComponent<RectTransform>();
            var renderer = GetComponent<Renderer>();
            int width = (int)rectTransform.rect.width;
            int height = (int)rectTransform.rect.height;
            videoTexture = new RenderTexture(width, height, 24);
            deviceName = devices[0].name;
            webCam = new WebCamTexture(deviceName);            
            renderer.material.mainTexture = webCam;
            webCam.Play();
            texture = new Texture2D(videoTexture.width, videoTexture.height, TextureFormat.RGB24, false);
        }
    }

    void Update() 
    {
        Graphics.Blit(webCam, videoTexture);
        Graphics.SetRenderTarget(videoTexture);
        texture.ReadPixels(new Rect(0, 0, videoTexture.width, videoTexture.height), 0, 0);
        texture.Apply();
        Graphics.SetRenderTarget(null);
        Debug.Log(texture);
        inferencer.Update(texture);
    }

    public void OnRenderObject() 
    {
        if (!inferencer.Initialized){ return; }

        bool debugHandLandmarks3D = true;
        if (debugHandLandmarks3D)
        { 
            var handLandmarks = inferencer.HandLandmarks;
            debugRenderer.DrawHand3D(handLandmarks);
        }
    }

    void OnDestroy(){ inferencer.Destroy(); }
}
