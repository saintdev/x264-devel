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

#define CL_CHECK(ret,func,...) do { \
    ret = func( __VA_ARGS__ ); \
    if ( err != CL_SUCCESS ) { \
        x264_log( h, X264_LOG_ERROR, "%s failed with error ID: %d!\n", #func, err ); \
        goto fail; \
    } \
} while(0)
#define CLFLAGS "-cl-mad-enable -cl-strict-aliasing"

void x264_opencl_frame_delete( x264_opencl_frame_t *opencl_frame )
{
    clReleaseMemObject( opencl_frame->plane[0] );
    clReleaseEvent( opencl_frame->uploaded[0] );

    for( int i = 0; i < MAX_PYRAMID_STEPS - 1; i++ ) {
        clReleaseMemObject( opencl_frame->lowres[i] );
        clReleaseEvent( opencl_frame->lowres_done[i] );
    }
    for( int i = 0; i < X264_BFRAME_MAX; i++ ) {
        for( int j = 0; j < 2; j++ ){
            clReleaseMemObject( opencl_frame->pmvs[j][i] );
            clReleaseEvent( opencl_frame->mvs_ready[j][i] );
        }
    }
}

int x264_opencl_frame_new( x264_t* h, x264_opencl_frame_t *opencl_frame )
{
    cl_int err = CL_SUCCESS;
    static const cl_image_format img_fmt = { CL_RGBA, CL_UNSIGNED_INT8 };

    /* FIXME: The buffer allocations should probably use stride instead of width. */
    if( h->opencl->b_image_support )
        CL_CHECK( opencl_frame->plane[0], clCreateImage2D, h->opencl->context, CL_MEM_READ_ONLY, &img_fmt, h->param.i_width, h->param.i_height, 0, NULL, &err );
    else
        CL_CHECK( opencl_frame->plane[0], clCreateBuffer, h->opencl->context, CL_MEM_READ_ONLY, h->param.i_width * h->param.i_height * sizeof( cl_uchar ), NULL, &err );

    for( int i = 0; i < MAX_PYRAMID_STEPS - 1; i++ ) {
        /* TODO: -According to the nVidia Programming Guide CL_MEM_ALLOC_HOST_POINTER
         *        is the only option that has a possibility of using pinned memory.
         *        This allows DMA when used with mapped buffers.
         *        *BENCHMARK*
         *       -Perhaps we should use two buffers here, one read only, and one write only.
         *        Then copy the downsampled frame from the write only buffer to the read only one.
         *        This would allow the possibility of using texture memory for lowres on cards that
         *        have read only texture memory.
         */
        if( h->opencl->b_image_support )
            CL_CHECK( opencl_frame->lowres[i], clCreateImage2D, h->opencl->context, CL_MEM_READ_WRITE, &img_fmt, h->param.i_width >> (1+i), h->param.i_height >> (1+i), 0, NULL, &err );
        else
            CL_CHECK( opencl_frame->lowres[i], clCreateBuffer, h->opencl->context, CL_MEM_READ_WRITE, (h->param.i_width * h->param.i_height * sizeof( cl_uchar )) >> (1+i), NULL, &err );
    }
    for( int i = 0; i < h->param.i_bframe + 1; i++ ) {
        for( int j = 0; j < 2; j++ )
            CL_CHECK( opencl_frame->pmvs[j][i], clCreateBuffer, h->opencl->context, CL_MEM_READ_WRITE, h->mb.i_mb_count * sizeof( cl_short2 ), NULL, &err );
    }


fail:
    return err;
}

static int x264_opencl_init_kernel_args( x264_t *h )
{
    cl_int err = CL_SUCCESS;

fail:
    return err;
}

static int x264_opencl_frame_upload( x264_t *h, x264_frame_t *fenc )
{
    cl_int err;
    static const size_t zero[3] = { 0 };
    const size_t region[3] = { fenc->i_width[0], fenc->i_lines[0], 1 };
    x264_opencl_frame_t *clfenc;

    // TODO: try to see if this can DMA.
    // This will likely require allocating two images, permanently mapping one
    // for CPU use, and copying to the GPU-only one

    if( fenc->opencl )
        return 0;

    /* Find the first unused opencl frame */
    for( int i = 0; i < X264_BFRAME_MAX + 3; i++ )
        if( !h->opencl->frames[i].i_ref_count ) {
            clfenc = &h->opencl->frames[i];
            break;
        }


    clReleaseEvent( clfenc->uploaded[0] );
    clfenc->uploaded[0] = NULL;

    for( int i = 0; i < MAX_PYRAMID_STEPS-1; i++ )
    {
        clReleaseEvent( clfenc->lowres_done[i] );
        clfenc->lowres_done[i] = NULL;
    }

    if( h->opencl->b_image_support )
        CL_CHECK( err, clEnqueueWriteImage, h->opencl->queue, clfenc->plane[0], CL_FALSE, zero, region, fenc->i_stride[0], 0, fenc->plane[0], 0, NULL, clfenc->uploaded[0] );
    else
        CL_CHECK( err, clEnqueueWriteBuffer, h->opencl->queue, clfenc->plane[0], CL_FALSE, 0, fenc->i_stride[0] * fenc->i_width[0] * sizeof( *fenc->plane[0] ), fenc->plane[0], 0, NULL, clfenc->uploaded[0] );

    fenc->opencl = clfenc;
    clfenc->i_ref_count++;

    return 1;

fail:
    return -1;
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
        x264_log( h, X264_LOG_ERROR, "Unable to find an OpenCL platform.\n" );

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
    cl_bool image_support = CL_FALSE;
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
    CL_CHECK( opencl->downsample_kernel, clCreateKernel, opencl->downsample_prog, "downsample_packed", &err );

    size = strlen(x264_opencl_simple_me_src);
    CL_CHECK( opencl->simple_me_prog, clCreateProgramWithSource, opencl->context, 1, &x264_opencl_simple_me_src, &size, &err );
    err = clBuildProgram( opencl->simple_me_prog, 0, NULL, CLFLAGS, NULL, NULL );
    if( err != CL_SUCCESS ) {
        x264_log( h, X264_LOG_ERROR, "clBuildProgram( simple_me ) failed with error ID: %d.\n", err );
        if( err == CL_BUILD_PROGRAM_FAILURE )
            x264_opencl_print_build_log( h, opencl->simple_me_prog, devices[0] );
        goto fail;
    }
    CL_CHECK( opencl->me_pyramid, clCreateKernel, opencl->simple_me_prog, "me_pyramid", &err );
    CL_CHECK( opencl->simple_me, clCreateKernel, opencl->simple_me_prog, "simple_me", &err );

    clUnloadCompiler();

    CL_CHECK( err, clGetDeviceInfo, devices[0], CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL );
    x264_log( h, X264_LOG_INFO, "using %s\n", device_name );
    CL_CHECK( err, clGetDeviceInfo, devices[0], CL_DEVICE_IMAGE_SUPPORT, sizeof( image_support ), &image_support, NULL );
    opencl->b_image_support = (image_support == CL_TRUE);

    /* FIXME: Should this be the number of refs or bframes? */
    for( int i = 0; i < h->param.i_bframe + 3; i++ )
        CL_CHECK( err, x264_opencl_frame_new, h, &opencl->frames[i] );

    CL_CHECK( err, x264_opencl_init_kernel_args, h );

    return 1;

fail:
    x264_free( opencl );
    return 0;
}

void x264_opencl_close( x264_t *h )
{
    for( int i = 0; i < X264_BFRAMES_MAX + 3; i++ )
        x264_opencl_frame_delete( &h->opencl->frames[i] );
    clReleaseKernel( h->opencl->simple_me );
    clReleaseKernel( h->opencl->me_pyramid );
    clReleaseProgram( h->opencl->simple_me_prog );
    clReleaseKernel( h->opencl->downsample_kernel );
    clReleaseProgram( h->opencl->downsample_prog );
    clReleaseCommandQueue( h->opencl->queue );
    clReleaseContext( h->opencl->context );

    x264_free( h->opencl );
}

static int x264_opencl_lowres_init( x264_t *h, x264_opencl_frame_t *fenc )
{
    /* FIXME: Multiple frames at once? */
    for( int i = 0; i < MAX_PYRAMID_STEPS-1; i++ ) {
        cl_mem *src = !i ? &fenc->plane[0] : &fenc->lowres[i-1];
        cl_event *src_done = !i ? &fenc->uploaded[0] : &fenc->lowres_done[i-1];
        size_t global_size[2] = { h->param.i_width >> i + 1, h->param.i_height >> i + 1 };
        /* FIXME: Ideal group sizes on current are 32 for nVidia, and 64 for ATI. It would
         *        be a good idea to detect what hardware we are on, and set group size
         *        accordingly.
         */
        size_t local_size[2] = { 32, 1 };
        clSetKernelArg( h->opencl->downsample_kernel, 0, sizeof(cl_mem), src );
        clSetKernelArg( h->opencl->downsample_kernel, 1, sizeof(cl_mem), &fenc->lowres[i] );
        clEnqueueNDRangeKernel( h->opencl->queue, h->opencl->downsample_kernel, 2, NULL, &global_size, &local_size, 1, src_done, &fenc->lowres_done[i] );
    }
    /* FIXME: Download the first downsampled image for non-OpenCL lookahead functions to use. */
}

int x264_opencl_analyse( x264_t *h )
{
    cl_int err = CL_SUCCESS;
    x264_frame_t *frames[X264_LOOKAHEAD_MAX+3] = { NULL, };
    int framecnt;
    int i_max_search = X264_MIN( h->lookahead->next.i_size, X264_LOOKAHEAD_MAX );
    if( h->param.b_deterministic )
        i_max_search = X264_MIN( i_max_search, h->lookahead->i_slicetype_length + !keyframe );

    /* FIXME: Should this be moved to x264_slicetype_analyse, a lot of duplicated code here. */
    if( !h->lookahead->last_nonb )
        return;
    frames[0] = h->lookahead->last_nonb;
    for( framecnt = 0; framecnt < i_max_search; framecnt++ ) {
        frames[framecnt+1] = h->lookahead->next.list[framecnt];
        if( x264_opencl_frame_upload( h, frames[framecnt+1] ) > 0 )
            CL_CHECK( err, x264_opencl_lowres_init, h, frames[framecnt+1]->opencl );
    }

    /* TODO: Do motion search on more than just the previous frame. */
    for( int i = 0; i < framecnt; i++ )
        CL_CHECK( err, x264_opencl_slicetype_cost, h, frames, i, 0, i+1 );

fail:
    return err;
}

x264_opencl_frame_unref( x264_t *h, int i_bframes )
{
    x264_opencl_frame_t *clframe;
    for( int i = 0; i < i_bframes; i++ ) {
        h->lookahead->next.list[i]->opencl->i_ref_count--;
        h->lookahead->next.list[i]->opencl = NULL;
    }
}
