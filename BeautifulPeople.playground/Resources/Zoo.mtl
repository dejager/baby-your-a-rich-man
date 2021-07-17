#include <metal_stdlib>
using namespace metal;

kernel void bigBrownBag(texture2d<float, access::write> o[[texture(0)]],
                        texture2d<float, access::read> i[[texture(1)]],
                        constant float &time [[buffer(0)]],
                        constant float2 *touchEvent [[buffer(1)]],
                        constant int &numberOfTouches [[buffer(2)]],
                        ushort2 gid [[thread_position_in_grid]]) {

  int width = o.get_width();
  int height = o.get_height();
  float2 res = float2(width, height);
  float2 p = float2(gid.xy);

  float2 xy = p / res.yy;

  float amp = 0.03;
  float freq = 10.0;
  float gray = 1.0;
  float div = 4.8 / res.y;
  float initialThickness = div * 0.2;

  const int patternCount = 6;

  float3 patterns[patternCount];
  patterns[0] = float3(-0.7071, 0.7071, 3.0);
  patterns[1] = float3(0.0, 1.0, 0.6);
  patterns[2] = float3(0.0, 1.0, 0.5);
  patterns[3] = float3(1.0, 0.0, 0.4);
  patterns[4] = float3(1.0, 0.0, 0.3);
  patterns[5] = float3(0.0, 1.0, 0.2);

  float4 color = float4(i.read(gid).rgb, 1.0);

  for(int i = 0; i < patternCount; i++) {
    float cosine = patterns[i].x;
    float sine = patterns[i].y;

    float2 point = float2(xy.x * cosine - xy.y * sine,
                          xy.x * sine + xy.y * cosine);

    float thickness = initialThickness * float(i + 1);
    float dist = fmod(point.y + thickness * 0.5 - sin(point.x * freq) * amp, div);
    float brillo = 0.3 * color.r + 0.4 * color.g + 0.3 * color.b;

    if(dist < thickness && brillo < 0.75 - 0.12 * float(i)) {
      float k = patterns[i].z;
      float x = (thickness - dist) / thickness;
      float fx = abs((x - 0.5) / k) - (0.5 - k) / k;
      gray = min(fx, gray);
    }
  }

  color = float4(gray, gray, gray, 1.0);

  o.write(color, gid);
}
