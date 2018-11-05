typedef struct tag_pixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Pixel;

__kernel void grayscale(__global Pixel* pixels, __global Pixel* outPixels)
{
    //int gid = 0;
    unsigned long gid = get_global_id(0);
    const int r = pixels[gid].r;
    const int g = pixels[gid].g;
    const int b = pixels[gid].b;
    const int max = r > g ? (r > b ? r : b) : (g > b ? g : b);
    const int min = r < g ? (r < b ? r : b) : (g < b ? g : b);
    const int gray = (max + min) / 2;
    outPixels[gid].r = gray;
    outPixels[gid].g = gray;
    outPixels[gid].b = gray;
}
