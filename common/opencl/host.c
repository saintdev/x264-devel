/*****************************************************************************
 * opencl.c: h264 encoder library (OpenCL host code)
 *****************************************************************************
 * Copyright (C) 2009 x264 project
 *
 * Authors: David Conrad <lessen42@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *****************************************************************************/

#include "common/common.h"
#include "opencl.h"

#define CHECK_CL(...) { __VA_ARGS__; if ( err ) return err; }
#define CLFLAGS "-cl-mad-enable -cl-strict-aliasing"

static void opencl_frame_delete( x264_frame_t *frame )
{
    int i;
    i = 0;
    for( i = 0; i < 3; i++ )
    {
        clReleaseMemObject( frame->opencl.plane[i] );
        clReleaseEvent( frame->opencl.uploaded[i] );
    }
    for( i = 0; i < MAX_PYRAMID_STEPS-1; i++ )
        clReleaseEvent( frame->opencl.lowres_done[i] );
}

static int opencl_frame_new( x264_t *h, x264_frame_t *frame, int b_fdec )
{
    static const cl_image_format img_fmt = { CL_RGBA, CL_UNSIGNED_INT8 };
    cl_int err;
    int i;

    if ( !b_fdec && h->frames.b_have_lowres )
    {
        // TODO: move this to x264_t
        frame->delete = opencl_frame_delete;

        for( i = 0; i < 3; i++ )
            CHECK_CL( frame->opencl.plane[i] = clCreateImage2D( h->opencl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &img_fmt, frame->i_width[i] >> 2, frame->i_lines[i], frame->i_stride[i], frame->plane[i], &err ) );

        for( i = 0; i < MAX_PYRAMID_STEPS-1; i++ )
            CHECK_CL( frame->opencl.lowres[i] = clCreateImage2D( h->opencl.context, CL_MEM_READ_WRITE, &img_fmt, frame->i_width[0] >> (3+i), frame->i_lines[0] >> (1+i), 0, NULL, &err ) );
    }

    return 0;
}

static int opencl_frame_upload( x264_t *h, x264_frame_t *fenc )
{
    cl_int err;
    int i;
    static const size_t zero[3] = { 0 };

    // TODO: try to see if this can DMA.
    // This will likely require allocating two images, permanently mapping one
    // for CPU use, and copying to the GPU-only one

    for( i = 0; i < 3; i++ )
    {
        clReleaseEvent( fenc->opencl.uploaded[i] );
        fenc->opencl.uploaded[i] = NULL;
    }
    for( i = 0; i < MAX_PYRAMID_STEPS-1; i++ )
    {
        clReleaseEvent( fenc->opencl.lowres_done[i] );
        fenc->opencl.lowres_done[i] = NULL;
    }

    for( i = 0; i < 3; i++ )
    {
        const size_t region[3] = { fenc->i_width[i] >> 2, fenc->i_lines[i], 1 };
        CHECK_CL( err = clEnqueueWriteImage( h->opencl.queue, fenc->opencl.plane[i], CL_FALSE, zero, region, fenc->i_stride[i], 0, fenc->plane[i], 0, NULL, &fenc->opencl.uploaded[i] ) );
    }

    for( i = 0; i < MAX_PYRAMID_STEPS-1; i++ )
    {
        const size_t work_size[2] = { fenc->i_width[0] >> (3+i), fenc->i_lines[0] >> (1+i) };
        cl_mem   src      = i ? fenc->opencl.lowres[i-1]      : fenc->opencl.plane[0];
        cl_event src_done = i ? fenc->opencl.lowres_done[i-1] : fenc->opencl.uploaded[0];

        CHECK_CL( err = clSetKernelArg( h->opencl.downsample_kernel, 0, sizeof(cl_mem), &fenc->opencl.lowres[i] ) );
        CHECK_CL( err = clSetKernelArg( h->opencl.downsample_kernel, 1, sizeof(cl_mem), &src ) );
        CHECK_CL( err = clEnqueueNDRangeKernel( h->opencl.queue, h->opencl.downsample_kernel, 2, NULL, work_size, NULL, 1, &src_done, &fenc->opencl.lowres_done[i] ) );

        CHECK_CL( err = clEnqueueReadImage( h->opencl.queue, fenc->opencl.lowres[i], 1, zero, work_size, fenc->i_stride[0], 0, fenc->plane[0], 1, &fenc->opencl.lowres_done[i], NULL ) );
    }
    return 0;
}


static void opencl_log( const char *errinfo, const void *priv, size_t cb, void *h )
{
    x264_log( h, X264_LOG_ERROR, "%s\n", errinfo );
}

// program sources
extern const char *x264_opencl_downsample_src;
extern const char *x264_opencl_simple_me_src;

int x264_opencl_init( x264_t *h )
{
    cl_int err;
    size_t size;
    cl_device_id devices[sizeof(cl_device_id) * 32];
    char device_name[1024];

    if( !h->param.b_opencl )
        return 0;

    /* FIXME: -We need to get a valid platform id to pass to CreateContextFromType instead of
     *         passing a NULL. This works on most implementations, but is undefined, and can
     *         break multi-platform systems.
     *        -Run only on CPU for now, this makes debugging kernels easier.
     */
    CHECK_CL( h->opencl.context = clCreateContextFromType( NULL, CL_DEVICE_TYPE_CPU, opencl_log, h, &err ) );

    // TODO: use device with max flops, and maybe create multiple queues to use multiple GPUs
    CHECK_CL( err = clGetContextInfo( h->opencl.context, CL_CONTEXT_DEVICES, sizeof(devices), devices, NULL ) );
    CHECK_CL( h->opencl.queue = clCreateCommandQueue( h->opencl.context, devices[0], 0, &err ) );

    size = strlen(x264_opencl_downsample_src);
    CHECK_CL( h->opencl.downsample_prog = clCreateProgramWithSource( h->opencl.context, 1, &x264_opencl_downsample_src, &size, &err ) );
    CHECK_CL( err = clBuildProgram( h->opencl.downsample_prog, 0, NULL, CLFLAGS, NULL, NULL ) );
    CHECK_CL( h->opencl.downsample_kernel = clCreateKernel( h->opencl.downsample_prog, "downsample_packed", &err ) );

    size = strlen(x264_opencl_simple_me_src);
    CHECK_CL( h->opencl.simple_me_prog = clCreateProgramWithSource( h->opencl.context, 1, &x264_opencl_simple_me_src, &size, &err ) );
    CHECK_CL( err = clBuildProgram( h->opencl.simple_me_prog, 0, NULL, CLFLAGS, NULL, NULL ) );
    CHECK_CL( h->opencl.me_pyramid = clCreateKernel( h->opencl.simple_me_prog, "me_pyramid", &err ) );
    CHECK_CL( h->opencl.me_full = clCreateKernel( h->opencl.simple_me_prog, "me_full", &err ) );

    CHECK_CL( err = clGetDeviceInfo( devices[0], CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL ) );
    x264_log( h, X264_LOG_INFO, "using %s\n", device_name );

    h->gpuf.frame_new = opencl_frame_new;
    h->gpuf.frame_upload = opencl_frame_upload;

    return 0;
}

void x264_opencl_close( x264_t *h )
{
    clReleaseKernel( h->opencl.downsample_kernel );
    clReleaseProgram( h->opencl.downsample_prog );
    clReleaseCommandQueue( h->opencl.queue );
    clReleaseContext( h->opencl.context );
}
