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

#define CL_CHECK(...) { __VA_ARGS__; if ( err != CL_SUCCESS ) goto fail; }
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
            CL_CHECK( frame->opencl.plane[i] = clCreateImage2D( h->opencl.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &img_fmt, frame->i_width[i] >> 2, frame->i_lines[i], frame->i_stride[i], frame->plane[i], &err ) );

        for( i = 0; i < MAX_PYRAMID_STEPS-1; i++ )
            CL_CHECK( frame->opencl.lowres[i] = clCreateImage2D( h->opencl.context, CL_MEM_READ_WRITE, &img_fmt, frame->i_width[0] >> (3+i), frame->i_lines[0] >> (1+i), 0, NULL, &err ) );
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
        CL_CHECK( err = clEnqueueWriteImage( h->opencl.queue, fenc->opencl.plane[i], CL_FALSE, zero, region, fenc->i_stride[i], 0, fenc->plane[i], 0, NULL, &fenc->opencl.uploaded[i] ) );
    }

    for( i = 0; i < MAX_PYRAMID_STEPS-1; i++ )
    {
        const size_t work_size[2] = { fenc->i_width[0] >> (3+i), fenc->i_lines[0] >> (1+i) };
        cl_mem   src      = i ? fenc->opencl.lowres[i-1]      : fenc->opencl.plane[0];
        cl_event src_done = i ? fenc->opencl.lowres_done[i-1] : fenc->opencl.uploaded[0];

        CL_CHECK( err = clSetKernelArg( h->opencl.downsample_kernel, 0, sizeof(cl_mem), &fenc->opencl.lowres[i] ) );
        CL_CHECK( err = clSetKernelArg( h->opencl.downsample_kernel, 1, sizeof(cl_mem), &src ) );
        CL_CHECK( err = clEnqueueNDRangeKernel( h->opencl.queue, h->opencl.downsample_kernel, 2, NULL, work_size, NULL, 1, &src_done, &fenc->opencl.lowres_done[i] ) );

        CL_CHECK( err = clEnqueueReadImage( h->opencl.queue, fenc->opencl.lowres[i], CL_TRUE, zero, work_size, fenc->i_stride[0], 0, fenc->plane[0], 1, &fenc->opencl.lowres_done[i], NULL ) );
    }
    return 0;
fail:
    return err;
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
    x264_opencl_t *opencl = NULL;
    CHECKED_MALLOCZERO( opencl, sizeof( *opencl ) );
    cl_int err = CL_SUCCESS;
    size_t size;
    cl_device_id devices[sizeof(cl_device_id) * 32];
    char device_name[1024];

    if( !h->param.b_opencl )
        goto fail;

    h->opencl = opencl;

    /* FIXME: -We need to get a valid platform id to pass to CreateContextFromType instead of
     *         passing a NULL. This works on most implementations, but is undefined, and can
     *         break multi-platform systems.
     *        -Run only on CPU for now, this makes debugging kernels easier.
     */
    CL_CHECK( opencl->context = clCreateContextFromType( NULL, CL_DEVICE_TYPE_CPU, opencl_log, h, &err ) );

    // TODO: use device with max flops, and maybe create multiple queues to use multiple GPUs
    CL_CHECK( err = clGetContextInfo( opencl->context, CL_CONTEXT_DEVICES, sizeof(devices), devices, NULL ) );
    CL_CHECK( opencl->queue = clCreateCommandQueue( opencl->context, devices[0], 0, &err ) );

    size = strlen(x264_opencl_downsample_src);
    CL_CHECK( opencl->downsample_prog = clCreateProgramWithSource( opencl->context, 1, &x264_opencl_downsample_src, &size, &err ) );
    /* TODO: Print build log if compilation fails. */
    CL_CHECK( err = clBuildProgram( opencl->downsample_prog, 0, NULL, CLFLAGS, NULL, NULL ) );
    CL_CHECK( opencl->downsample_kernel = clCreateKernel( opencl->downsample_prog, "downsample_packed", &err ) );

    size = strlen(x264_opencl_simple_me_src);
    CL_CHECK( opencl->simple_me_prog = clCreateProgramWithSource( opencl->context, 1, &x264_opencl_simple_me_src, &size, &err ) );
    /* TODO: Print build log if compilation fails. */
    CL_CHECK( err = clBuildProgram( opencl->simple_me_prog, 0, NULL, CLFLAGS, NULL, NULL ) );
    CL_CHECK( opencl->me_pyramid = clCreateKernel( opencl->simple_me_prog, "me_pyramid", &err ) );
    CL_CHECK( opencl->me_full = clCreateKernel( opencl->simple_me_prog, "me_full", &err ) );

    CL_CHECK( err = clUnloadCompiler() );

    CL_CHECK( err = clGetDeviceInfo( devices[0], CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL ) );
    x264_log( h, X264_LOG_INFO, "using %s\n", device_name );

    h->gpuf.frame_new = opencl_frame_new;
    h->gpuf.frame_upload = opencl_frame_upload;

    /* FIXME: Just using the same number of frames in our list as lookahead.
     *        This may not be optimal.
     */
    if( x264_synch_frame_list_init( &opencl->ifbuf, h->param.i_sync_lookahead+3 ) ||
        x264_synch_frame_list_init( &opencl->next, h->frames.i_delay+3 ) ||
        x264_synch_frame_list_init( &opencl->ofbuf, h->frames.i_delay+3 ) )
        goto fail;

    x264_t *opencl_h = h->thread[h->param.i_threads + 1];
    *opencl_h = *h;

    if( x264_pthread_create( &opencl_h->thread_handle, NULL, (void *)x264_opencl_thread, opencl_h ) )
        goto fail;

    opencl->b_thread_active = 1;

    return 0;
fail:
    x264_free( opencl );
    if( err != CL_SUCCESS )
        x264_log( h, X264_LOG_ERROR, "OpenCL error ID: %d", err );
    return -1
}

void x264_opencl_close( x264_t *h )
{
    x264_pthread_mutex_lock( &h->opencl->ifbuf.mutex );
    h->opencl->b_exit_thread = 1;
    x264_pthread_cond_broadcast( &h->opencl->ifbuf.cv_fill );
    x264_pthread_mutex_unlock( &h->opencl->ifbuf.mutex );
    x264_pthread_join( h->thread[h->param.i_threads + 1]->thread_handle, NULL );
    x264_free( h->thread[h->param.i_threads + 1] );

    clReleaseKernel( h->opencl->me_simple );
    clReleaseKernel( h->opencl->me_pyramid );
    clReleaseProgram( h->opencl->simple_me_prog );
    clReleaseKernel( h->opencl.downsample_kernel );
    clReleaseProgram( h->opencl.downsample_prog );
    clReleaseCommandQueue( h->opencl.queue );
    clReleaseContext( h->opencl.context );

    x264_free( h->opencl );
}

void x264_opencl_put_frame( x264_t *h, x264_frame_t *frame )
{
    x264_synch_frame_list_push( &h->opencl->ifbuf, frame );
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
        x264_synch_frame_list_shift( &h->opencl->ofbuf, &h->openc->ifbuf, 1 );
        x264_pthread_mutex_unlock( &h->opencl->ifbuf.mutex );
        x264_pthread_mutex_unlock( &h->opencl->ofbuf.mutex );
    }   /* end of input frames */
}
