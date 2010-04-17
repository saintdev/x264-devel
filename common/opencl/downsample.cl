/*****************************************************************************
 * downsample.cl: h264 encoder library (OpenCL downsampling)
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

const sampler_t s = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

static inline uint4 rnd_avg32( uint4 a, uint4 b )
{
    return (a | b) - (((a ^ b) & (uint4)(0xfefefefe)) >> 1);
}

#define FILTER(a,b,c,d) rhadd(rhadd(a,b), rhadd(c,d))
// more readable imo but nvidia's implementation chokes on it
//#define FILTER(a,b,c,d) (rhadd(rhadd(a,b), rhadd(c,d)))

#define FILTER2(a,b,c,d) ((a + b + c + d + 2) >> 2)
#define FILTER_SWAR(a,b,c,d) (rnd_avg32(rnd_avg32(a,b), rnd_avg32(c,d)))

// todo: hpel planes? worth doing in floating point and using FILTER_LINEAR?

// for CL_R, UINT8
kernel void downsample_simple( write_only image2d_t dst, read_only image2d_t src )
{
    int2 loc = (int2)(get_global_id(0), get_global_id(1));

    uint pix = FILTER(read_imageui( src, s, loc*2               ).s0,
                      read_imageui( src, s, loc*2 + (int2)(0,1) ).s0,
                      read_imageui( src, s, loc*2 + (int2)(1,0) ).s0,
                      read_imageui( src, s, loc*2 + (int2)(1,1) ).s0);
    write_imageui( dst, loc, (uint4)(pix) );
}

// for CL_RGBA, UINT8
kernel void downsample_packed( write_only image2d_t dst, read_only image2d_t src )
{
    int2 loc = (int2)(get_global_id(0), get_global_id(1));

    // uchar4 + rhadd would be the most valid and in theory let the most
    // efficient code to be generated, but requires convert_uchar4() etc.
    // and in practice generates worse code for what we care about (GPUs)
    uint4 p00 = read_imageui( src, s, loc*2               );
    uint4 p10 = read_imageui( src, s, loc*2 + (int2)(0,1) );
    uint4 p01 = read_imageui( src, s, loc*2 + (int2)(1,0) );
    uint4 p11 = read_imageui( src, s, loc*2 + (int2)(1,1) );

    uint4 a = (uint4)(p00.s02, p01.s02);
    uint4 b = (uint4)(p00.s13, p01.s13);
    uint4 c = (uint4)(p10.s02, p11.s02);
    uint4 d = (uint4)(p10.s13, p11.s13);

    write_imageui( dst, loc, FILTER(a, b, c, d) );
}
