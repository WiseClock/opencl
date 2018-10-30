#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
extern "C"
{
    #include "lib/jpeglib.h"
}
#ifdef __APPLE__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif
#include "cl.hpp"

using namespace std;

struct Pixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

int grayscaleFilter(const Pixel* pixels, Pixel*& newPixels, const unsigned long int length)
{
    for (unsigned long int i = 0; i < length; ++i)
    {
        const int r = pixels[i].r;
        const int g = pixels[i].g;
        const int b = pixels[i].b;

        const int gray = (max(r, max(g, b)) + min(r, min(g, b))) / 2;
        // another way but faster:
        // const double gray_d = (r * 0.3 + g * 0.59 + b * 0.11);
        // const int gray = (int)(gray_d + 0.5);

        newPixels[i].r = gray;
        newPixels[i].g = gray;
        newPixels[i].b = gray;
    }
    return 0;
}

int readImage(const char* name, struct Pixel*& pixels, unsigned long int& width, unsigned long int& height)
{
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    // read jpeg file
    FILE *file;
    if ((file = fopen(name, "rb")) == NULL)
        return -1;

    // read jpeg header (width and height)
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, file);
    (void) jpeg_read_header(&cinfo, (boolean)true);
    (void) jpeg_start_decompress(&cinfo);
    width = cinfo.output_width;
    height = cinfo.output_height;

    unsigned long int row_stride = width * cinfo.output_components;
    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray) ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);
    pixels = (Pixel*)malloc(width * height * sizeof(Pixel));

    // process image line by line
    unsigned long int pixelCounter = 0;
    while (cinfo.output_scanline < cinfo.output_height)
    {
        (void) jpeg_read_scanlines(&cinfo, buffer, 1);
        // process line pixels
        for (int i = 0; i < width; ++i)
        {
            pixels[pixelCounter].r = buffer[0][cinfo.output_components * i];
            if (cinfo.output_components > 2)
            {
                pixels[pixelCounter].g = buffer[0][cinfo.output_components * i + 1];
                pixels[pixelCounter].b = buffer[0][cinfo.output_components * i + 2];
            }
            else
            {
                pixels[pixelCounter].g = pixels[pixelCounter].r;
                pixels[pixelCounter].b = pixels[pixelCounter].r;
            }
            pixelCounter++;
        }
    }

    fclose(file);
    (void) jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return 0;
}

int writeImage(const char* name, const struct Pixel* pixels, const unsigned long int width, const unsigned long int height)
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    // write to file
    FILE *file;
    if ((file = fopen(name, "wb")) == NULL)
    {
        cerr << "Can't open output file." << endl;
        return -1;
    }

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, file);

    cinfo.image_width = width;  
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_start_compress(&cinfo, (boolean)true);

    unsigned long int row_stride = width * cinfo.input_components;
    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray) ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

    unsigned long int counter = 0;
    while(cinfo.next_scanline < cinfo.image_height)
    {
        for (int i = 0; i < width; ++i)
        {
            buffer[0][cinfo.input_components * i] = pixels[counter].r;
            buffer[0][cinfo.input_components * i + 1] = pixels[counter].g;
            buffer[0][cinfo.input_components * i + 2] = pixels[counter].b;
            counter++;
        }
        jpeg_write_scanlines(&cinfo, buffer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(file);

    cout << "Image saved." << endl;
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <file_name.jpg>" << endl;
        return -1;
    }

    struct Pixel* pixels;
    unsigned long int width;
    unsigned long int height;
    
    if (readImage(argv[1], pixels, width, height) == -1)
    {
        cerr << "Invalid image file." << endl;
        return -1;
    }

    cout << "Image reading completed." << endl;

    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0)
    {
        cout << "No valid OpenCL platform." << endl;
        return -1;
    }
    cl::Platform platform = platforms.front();
    vector<cl::Device> clDevicesCPU;
    vector<cl::Device> clDevicesGPU;
    platform.getDevices(CL_DEVICE_TYPE_CPU, &clDevicesCPU);
    platform.getDevices(CL_DEVICE_TYPE_GPU, &clDevicesGPU);
    if (clDevicesCPU.size() == 0 && clDevicesGPU.size() == 0)
    {
        cout << "No valid OpenCL device." << endl;
        return -1;
    }

    fstream clFile("main.cl");
    string clSrc(istreambuf_iterator<char>(clFile), (istreambuf_iterator<char>()));
    cl::Program::Sources sources(1, make_pair(clSrc.c_str(), clSrc.length() + 1));

    bool loop = true;
    int selection;
    while (loop)
    {
        cout << endl;
        cout << "0. Exit program." << endl;
        cout << "1. Serial." << endl;
        cout << "2. OpenCL CPU." << endl;
        cout << "3. OpenCL GPU." << endl;
        cout << "4. OpenCL CPU + GPU." << endl;
        cout << "Please select: ";
        cin >> selection;

        const int selMin = 0;
        const int selMax = 4;
        const int sel = selection < selMin ? selMin : selection > selMax ? selMax : selection;
        switch (sel)
        {
            case 0:
            {
                cout << "Bye." << endl;
                loop = false;
                break;
            }
            case 1:
            {
                // serial
                chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
                struct Pixel* newPixels = (Pixel*)malloc(width * height * sizeof(Pixel));;
                grayscaleFilter(pixels, newPixels, width * height);
                chrono::high_resolution_clock::time_point finish = std::chrono::high_resolution_clock::now();
                double elapsed = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / (double)1000000;
                cout << endl << "Serial elapsed time: " << elapsed << " ms" << endl;
                writeImage("out.jpg", newPixels, width, height);
                delete newPixels;
                break;
            }
            case 2:
            {
                // OpenCL CPU
                if (clDevicesCPU.size() == 0)
                {
                    cerr << "No available OpenCL CPU device." << endl;
                    return -1;
                }
                cl::Device device = clDevicesCPU.front();
                cl::Context context(device);
                cl::Program program(context, sources);
                auto err = program.build("-cl-std=CL1.2");

                cl::Buffer clBuff(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(Pixel) * width * height, pixels);
                cl::Buffer clOutBuff(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(Pixel) * width * height);
                cl::Kernel kernel(program, "grayscale", &err);
                kernel.setArg(0, clBuff);
                kernel.setArg(1, clOutBuff);

                cl::CommandQueue queue(context, device);

                chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();

                queue.enqueueNDRangeKernel(kernel, 0, cl::NDRange(width * height));
                struct Pixel* newPixels = (Pixel*)malloc(width * height * sizeof(Pixel));
                queue.enqueueReadBuffer(clOutBuff, CL_FALSE, 0, sizeof(Pixel) * width * height, newPixels);
                queue.finish();

                chrono::high_resolution_clock::time_point finish = std::chrono::high_resolution_clock::now();
                double elapsed = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / (double)1000000;
                cout << endl << "OpenCL CPU elapsed time: " << elapsed << " ms" << endl;

                writeImage("out.jpg", newPixels, width, height);
                delete newPixels;
                break;
            }
            case 3:
            {
                // OpenCL GPU
                if (clDevicesGPU.size() == 0)
                {
                    cerr << "No available OpenCL GPU device." << endl;
                    return -1;
                }
                cl::Device device = clDevicesGPU.front();
                cl::Context context(device);
                cl::Program program(context, sources);
                auto err = program.build("-cl-std=CL1.2");

                cl::Buffer clBuff(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(Pixel) * width * height, pixels);
                cl::Buffer clOutBuff(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(Pixel) * width * height);
                cl::Kernel kernel(program, "grayscale", &err);
                kernel.setArg(0, clBuff);
                kernel.setArg(1, clOutBuff);

                cl::CommandQueue queue(context, device);

                chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();

                queue.enqueueNDRangeKernel(kernel, 0, cl::NDRange(width * height));
                struct Pixel* newPixels = (Pixel*)malloc(width * height * sizeof(Pixel));
                queue.enqueueReadBuffer(clOutBuff, CL_FALSE, 0, sizeof(Pixel) * width * height, newPixels);
                queue.finish();

                chrono::high_resolution_clock::time_point finish = std::chrono::high_resolution_clock::now();
                double elapsed = chrono::duration_cast<chrono::nanoseconds>(finish - start).count() / (double)1000000;
                cout << endl << "OpenCL GPU elapsed time: " << elapsed << " ms" << endl;

                writeImage("out.jpg", newPixels, width, height);
                delete newPixels;
                break;
            }
            case 4:
            {
                cerr << "Not implemented." << endl;
                break;
            }
        }
    }

    return 0;
}
