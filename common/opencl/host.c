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

#define CL_CHECK(ret,func,...) { \
    ret = func( __VA_ARGS__ ); \
    if ( err != CL_SUCCESS ) { \
        x264_log( h, X264_LOG_ERROR, "%s failed with error ID: %d!\n", #func, err ); \
        goto fail; \
    } \
}
#define CLFLAGS "-cl-mad-enable -cl-strict-aliasing"

void x264_opencl_frame_delete( x264_opencl_frame_t *opencl_frame )
{
    int i;
    i = 0;
    for( i = 0; i < 3; i++ )
    {
        clReleaseMemObject( opencl_frame->plane[i] );
        clReleaseEvent( opencl_frame->uploaded[i] );
    }
    for( i = 0; i < MAX_PYRAMID_STEPS-1; i++ ) {
        clReleaseMemObject( opencl_frame->lowres[i] );
        clReleaseEvent( opencl_frame->lowres_done[i] );
    }

    x264_free( opencl_frame );
}

int x264_opencl_frame_new( x264_opencl_t *opencl, x264_frame_t *frame, int b_fdec )
{
    x264_opencl_frame_t *opencl_frame;
    CHECKED_MALLOCZERO( opencl_frame, sizeof( *opencl_frame ) );
    frame->opencl = opencl_frame;
    static const cl_image_format img_fmt = { CL_RGBA, CL_UNSIGNED_INT8 };
    cl_int err;
    int i;

    if ( !b_fdec )
    {
        /* TODO: According to nVidia docs CL_MEM_ALLOC_HOST_POINTER is best to use.
         *       *BENCHMARK*
         */
        for( i = 0; i < 3; i++ )
            CL_CHECK( opencl_frame->plane[i], clCreateImage2D, opencl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &img_fmt, frame->i_width[i] >> 2, frame->i_lines[i], frame->i_stride[i], frame->plane[i], &err );

        for( i = 0; i < MAX_PYRAMID_STEPS-1; i++ )
            CL_CHECK( opencl_frame->lowres[i], clCreateImage2D, opencl->context, CL_MEM_READ_WRITE, &img_fmt, frame->i_width[0] >> (3+i), frame->i_lines[0] >> (1+i), 0, NULL, &err );
    }

    return 0;

fail:
    return err;
}

static int x264_opencl_frame_upload( x264_t *h, x264_frame_t *fenc )
{
    cl_int err;
    int i;
    static const size_t zero[3] = { 0 };

    // TODO: try to see if this can DMA.
    // This will likely require allocating two images, permanently mapping one
    // for CPU use, and copying to the GPU-only one

    for( i = 0; i < 3; i++ )
    {
        clReleaseEvent( fenc->opencl->uploaded[i] );
        fenc->opencl->uploaded[i] = (cl_event)NULL;
    }
    for( i = 0; i < MAX_PYRAMID_STEPS-1; i++ )
    {
        clReleaseEvent( fenc->opencl->lowres_done[i] );
        fenc->opencl->lowres_done[i] = NULL;
    }

    for( i = 0; i < 3; i++ )
    {
        const size_t region[3] = { fenc->i_width[i] >> 2, fenc->i_lines[i], 1 };
        CL_CHECK( err, clEnqueueWriteImage, h->opencl->queue, fenc->opencl->plane[i], CL_FALSE, zero, region, fenc->i_stride[i], 0, fenc->plane[i], 0, NULL, &fenc->opencl->uploaded[i] );
    }

    for( i = 0; i < MAX_PYRAMID_STEPS-1; i++ )
    {
        const size_t work_size[2] = { fenc->i_width[0] >> (3+i), fenc->i_lines[0] >> (1+i) };
        cl_mem   src      = i ? fenc->opencl->lowres[i-1]      : fenc->opencl->plane[0];
        cl_event src_done = i ? fenc->opencl->lowres_done[i-1] : fenc->opencl->uploaded[0];

        CL_CHECK( err, clSetKernelArg, h->opencl->downsample_kernel, 0, sizeof(cl_mem), &fenc->opencl->lowres[i] );
        CL_CHECK( err, clSetKernelArg, h->opencl->downsample_kernel, 1, sizeof(cl_mem), &src );
        CL_CHECK( err, clEnqueueNDRangeKernel, h->opencl->queue, h->opencl->downsample_kernel, 2, NULL, work_size, NULL, 1, &src_done, &fenc->opencl->lowres_done[i] );

        CL_CHECK( err, clEnqueueReadImage, h->opencl->queue, fenc->opencl->lowres[i], CL_TRUE, zero, work_size, fenc->i_stride[0], 0, fenc->plane[0], 1, &fenc->opencl->lowres_done[i], NULL );
    }
    return 0;
fail:
    return err;
}

static void x264_opencl_thread( x264_t *h )
{
    int shift;
    while( !h->opencl->b_exit_thread )
    {
        x264_pthread_mutex_lock( &h->opencl->ifbuf.mutex );
        while( !h->opencl->ifbuf.i_size && !h->opencl->b_exit_thread )
            x264_pthread_cond_wait( &h->opencl->ifbuf.cv_fill, &h->opencl->ifbuf.mutex );
        x264_pthread_mutex_unlock( &h->opencl->ifbuf.mutex );

        x264_opencl_frame_upload( h, h->opencl->ifbuf.list[0] );

        x264_pthread_mutex_lock( &h->opencl->ofbuf.mutex );
        while( h->opencl->ofbuf.i_size == h->opencl->ofbuf.i_max_size )
            x264_pthread_cond_wait( &h->opencl->ofbuf.cv_empty, &h->opencl->ofbuf.mutex );

        x264_pthread_mutex_lock( &h->opencl->ifbuf.mutex );
        x264_synch_frame_list_shift( &h->opencl->ofbuf, &h->opencl->ifbuf, 1 );
        x264_pthread_mutex_unlock( &h->opencl->ifbuf.mutex );
        x264_pthread_mutex_unlock( &h->opencl->ofbuf.mutex );
    }   /* end of input frames */
}

static void opencl_log( const char *errinfo, const void *priv, size_t cb, void *h )
{
    x264_log( h, X264_LOG_ERROR, "%s\n", errinfo );
}

static cl_int x264_opencl_get_platform( x264_t *h, cl_platform_id *platform )
{
    cl_int err = CL_SUCCESS;
    cl_uint count;

    CL_CHECK( err, clGetPlatformIDs, 0, NULL, &count );
    if( count > 0 )
    {
        cl_platform_id *platforms;
        CHECKED_MALLOC( platforms, count * sizeof( *platforms ) );
        CL_CHECK( err, clGetPlatformIDs, count, platforms, NULL );
        /* TODO: Intelligently select the best (all?) platform */
        *platform = platforms[0];
        x264_free( platforms );
    }
    else
        goto fail;

    return CL_SUCCESS;

fail:
    return err;
}

static void x264_opencl_print_build_log( x264_t *h, cl_program program, cl_device_id device )
{
    cl_int err;
    char *log;
    size_t log_size;

    CL_CHECK( err, clGetProgramBuildInfo, program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size );
    CHECKED_MALLOC( log, log_size );
    CL_CHECK( err, clGetProgramBuildInfo, program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL );
    x264_log( h, X264_LOG_ERROR, "%s\n", log );

fail:
    x264_free( log );
    return;
}

// program sources
extern const char *x264_opencl_downsample_src;
extern const char *x264_opencl_simple_me_src;

int x264_opencl_init( x264_t *h )
{
    x264_opencl_t *opencl = NULL;
    CHECKED_MALLOCZERO( opencl, sizeof( *opencl ) );
    cl_int err = CL_SUCCESS;
    size_t size;
    cl_device_id devices[sizeof(cl_device_id) * 32];
    char device_name[1024];
    const cl_context_properties props[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)opencl->platform,
        0
    };

    h->opencl = opencl;

    CL_CHECK( err, x264_opencl_get_platform, h, &opencl->platform );
    CL_CHECK( opencl->context, clCreateContextFromType, props, CL_DEVICE_TYPE_GPU, opencl_log, h, &err );

    // TODO: use device with max flops, and maybe create multiple queues to use multiple GPUs
    CL_CHECK( err, clGetContextInfo, opencl->context, CL_CONTEXT_DEVICES, sizeof(devices), devices, NULL );
    CL_CHECK( opencl->queue, clCreateCommandQueue, opencl->context, devices[0], 0, &err );

    size = strlen(x264_opencl_downsample_src);
    CL_CHECK( opencl->downsample_prog, clCreateProgramWithSource, opencl->context, 1, &x264_opencl_downsample_src, &size, &err );
    err = clBuildProgram( opencl->downsample_prog, 0, NULL, CLFLAGS, NULL, NULL );
    if( err != CL_SUCCESS ) {
        x264_log( h, X264_LOG_ERROR, "clBuildProgram( downsample ) failed with error ID: %d.\n", err );
        if( err == CL_BUILD_PROGRAM_FAILURE )
            x264_opencl_print_build_log( h, opencl->downsample_prog, devices[0] );
        goto fail;
    }
    CL_CHECK( opencl->downsample_kernel, clCreateKernel opencl->downsample_prog, "downsample_packed", &err );

    size = strlen(x264_opencl_simple_me_src);
    CL_CHECK( opencl->simple_me_prog, clCreateProgramWithSource, opencl->context, 1, &x264_opencl_simple_me_src, &size, &err );
    err = clBuildProgram( opencl->simple_me_prog, 0, NULL, CLFLAGS, NULL, NULL );
    if( err != CL_SUCCESS ) {
        x264_log( h, X264_LOG_ERROR, "clBuildProgram( simple_me ) failed with error ID: %d.\n", err );
        if( err == CL_BUILD_PROGRAM_FAILURE )
            x264_opencl_print_build_log( h, opencl->simple_me_prog, devices[0] );
        goto fail;
    }
    CL_CHECK( opencl->me_pyramid, clCreateKernel( opencl->simple_me_prog, "me_pyramid", &err );
    CL_CHECK( opencl->me_full, clCreateKernel, opencl->simple_me_prog, "me_full", &err );

    clUnloadCompiler();

    CL_CHECK( err, clGetDeviceInfo, devices[0], CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL );
    x264_log( h, X264_LOG_INFO, "using %s\n", device_name );

    return 0;

fail:
    x264_free( opencl );
    return -1;
}

void x264_opencl_close( x264_t *h )
{
    clReleaseKernel( h->opencl->me_full );
    clReleaseKernel( h->opencl->me_pyramid );
    clReleaseProgram( h->opencl->simple_me_prog );
    clReleaseKernel( h->opencl->downsample_kernel );
    clReleaseProgram( h->opencl->downsample_prog );
    clReleaseCommandQueue( h->opencl->queue );
    clReleaseContext( h->opencl->context );

    x264_free( h->opencl );
}

void x264_opencl_put_frame( x264_t *h, x264_frame_t *frame )
{
    x264_synch_frame_list_push( &h->opencl->ifbuf, frame );
}
