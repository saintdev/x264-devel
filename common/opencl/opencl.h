/*****************************************************************************
 * opencl.h: h264 encoder library (OpenCL host code)
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

#ifndef X264_OPENCL_H
#define X264_OPENCL_H

#ifdef SYS_MACOSX
#    include <OpenCL/opencl.h>
#else
#    include <CL/cl.h>
#endif

#define MAX_PYRAMID_STEPS 4

struct x264_opencl
{
    cl_platform_id  platform;
    cl_context      context;

    cl_program      downsample_prog;
    cl_kernel       downsample_kernel;

    cl_program      simple_me_prog;
    cl_kernel       me_pyramid;
    cl_kernel       me_full;

    // TODO: one per thread is safe, but Apple warns that having a command queue for
    // each thread may be more expensive than implementing locking on a single one
    cl_command_queue queue;
};

struct x264_opencl_frame
{
    cl_mem      plane[3];
    cl_event    uploaded[3];    // clEnqueueWriteImage has completed
    cl_mem      lowres[MAX_PYRAMID_STEPS-1];
    cl_event    lowres_done[MAX_PYRAMID_STEPS-1];
};

int  x264_opencl_init( x264_t *h );
void x264_opencl_close( x264_t *h );
void x264_opencl_put_frame( x264_t *h, x264_frame_t *frame );
int  x264_opencl_frame_new( struct x264_opencl *opencl, x264_frame_t *frame, int b_fdec );
void x264_opencl_frame_delete( x264_opencl_frame_t *opencl_frame );

#endif
