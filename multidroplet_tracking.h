#pragma once

extern "C" __declspec(dllexport) void openCamera(int cameraIndex = 0, double width = 800, double height = 600, double fps = 30);
extern "C" __declspec(dllexport) void setCameraParams(double brightness, double contrast, double saturation, double hue, double gain, double exposure);
extern "C" __declspec(dllexport) void openVideo(char* path);
extern "C" __declspec(dllexport) void initTrackers();
extern "C" __declspec(dllexport) int* updateTrackers();
extern "C" __declspec(dllexport) void closeTrackers();  // Must be closed manually
