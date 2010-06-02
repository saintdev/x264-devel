/*****************************************************************************
 * downsample.cl: h264 encoder library (OpenCL pyramidal motion estimation)
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

#include "common/opencl/common.hl"
#include "common/opencl/images.hl"


typedef struct {
    short2  mv;
    uint    cost;
} x264_opencl_result_t;

#define COPY2_IF_LT(x,y,a,b)\
if((y)<(x))\
{\
    (x)=(y);\
    (a)=(b);\
}

#define COPY1_IF_LT(x,y)\
if((y)<(x))\
    (x)=(y);

// some thoughts:
// ME by its nature cannot do coalesced loads, thus it must use textures to avoid major slowdowns
// cause by uncooalesced loads.
// NOTE: texture reads are probably as fast as shared reads, so there shouldn't be any gain
// from loading an area into shared memory to reduce texture reads.

inline uint simple_sad( x264_image_t pix1, x264_image_t pix2, uint2 block_pos, short2 mv )
{
    uint sum = 0;
    int2 pos;

    for( pos.y = block_pos.y; pos.y < block_pos.y + 16; pos.y++ )
        for( pos.x = block_pos.x; pos.x < block_pos.x + 16; pos.x++ ) {
            uint a = x264_read_image( pix1, s, pos );
            uint b = x264_read_image( pix2, s, pos + (int2)(mv) );
            sum += abs_diff( a, b );
        }
    return sum;
}

inline int2 median_mv( int2 a, int2 b, int2 c )
{
    int2 t = a;
    a = max( a, b );
    b = min( t, b );
    a = min( c, a );
    a = max( b, a );
    return a;
}

__kernel void simple_me( READ_ONLY x264_image_t fenc, READ_ONLY x264_image_t fref,
                         global x264_opencl_result_t *result, uint scale_factor,
                         uint stride, uint mb_stride )
{
    const uint2 block = (int2)(get_global_id(0), get_global_id(1));
    const uint2 block_position = block * (uint2)(16);

    const x264_opencl_result_t pred_l  = result[(block.x-1 + block.y     * mb_stride) >> 1];
    const x264_opencl_result_t pred_t  = result[(block.x   + (block.y-1) * mb_stride) >> 1];
    const x264_opencl_result_t pred_tr = result[(block.x+1 + (block.y-1) * mb_stride) >> 1];

    // Compute the median of the three vectors (multiple of 2 because the blocks were down sampled)
    const short2 pmv = median_mv(pred_l.mv, pred_t.mv, pred_tr.mv) * (short2)(2);
/*
    int4 motion_vector;
    motion_vector.xy = read_imagei(previous, sampler_motion_vector, block / 2).xy * 2;*/

    // Compute the best starting point
    motion_vector.z = compute_motion_cost(imageA, imageB, sampler, block_position, motion_vector.xy, motion_vector_predicted, scale_factor);
    int4 candidate;
    candidate.xy = (0,0);
    candidate.z = compute_motion_cost(imageA, imageB, sampler, block_position, candidate.xy, motion_vector_predicted, scale_factor);
    motion_vector = take_best_motion(candidate, motion_vector);
    candidate.xy = motion_vector_predicted;
    candidate.z = compute_motion_cost(imageA, imageB, sampler, block_position, candidate.xy, motion_vector_predicted, scale_factor);
    motion_vector = take_best_motion(candidate, motion_vector);

    for(int iter = 0 ; iter < 4 ; ++iter)
    {
        motion_vector = find_best_hexa_motion(imageA, imageB, sampler, block_position, motion_vector, motion_vector_predicted, scale_factor);
    }

    // do a local search
    const int2 starting = motion_vector.xy;
    for(int x = -1 ; x < 1 ; ++x)
        for(int y = -1 ; y < 1 ; ++y)
        {
            int4 candidate;
            candidate.xy = starting + (x,y);
            candidate.z = compute_motion_cost(imageA, imageB, sampler, block_position, candidate.xy, motion_vector_predicted, scale_factor);
            motion_vector = take_best_motion(candidate, motion_vector);
        }

    write_imagei(result, block, (int4)(motion_vector.x, motion_vector.y, 0, 0));
}

uint vec_sad_aligned( image2d_t fenc, image2d_t fref, int2 mb, int2 mv, int size )
{
    uint sum = 0;
    int2 pos;

    mv.x >>= 2;

    for (pos.y = mb.y; pos.y < mb.y + size; pos.y++)
        for (pos.x = mb.x; pos.x < mb.x + size>>2; pos.x++)
        {
            uint4 a = read_imageui(fenc, s, pos);
            uint4 b = read_imageui(fref, s, pos + mv);

            uint4 diff = abs_diff(a, b);
            sum += diff.s0 + diff.s1 + diff.s2 + diff.s3;
        }
    return sum;
}

// texture sampling always reads all four channels, which implies 4-byte aligned reads
// are the extra bitops needed to deal with this worth the texture fetches saved?
uint vec_sad_unaligned( image2d_t fenc, image2d_t fref, int2 mb, int2 mv, int size )
{
    uint sum = 0;
    uint align = mv.x & 3;
    int2 pos;

    mv.x >>= 2;

    for (pos.y = mb.y; pos.y < mb.y + size; pos.y++)
        for (pos.x = mb.x; pos.x < mb.x + size>>2; pos.x++)
        {
            uint4 a  = read_imageui(fenc, s, pos);
            uint4 b1 = read_imageui(fref, s, pos + mv);
            uint4 b2 = read_imageui(fref, s, pos + mv + (int2)(1,0));

            // texture fetches leave each channel in its own register
            // so pack one register before extracting the needed 4 bytes
            // todo: SWAR
            uint b1p = (b1.s0 << 24) | (b1.s1 << 16) | (b1.s2 << 8) | b1.s3;
            uint b2p = (b2.s0 << 24) | (b2.s1 << 16) | (b2.s2 << 8) | b2.s3;
            uint bp = (b1p << align*8) | (b2p >> (3-align)*8);
            uint4 b = (uint4)(bp>>24, (bp>>16)&0xff, (bp>>8)&0xff, bp&0xff);

            uint4 diff = abs_diff(a, b);
            sum += diff.s0 + diff.s1 + diff.s2 + diff.s3;
        }
    return sum;
}

//__constant int2 diamond[2][2] =
constant int2 diamond[2][2] =
{
    {(int2)(-1,0), (int2)(0, 1)},
    {(int2)( 1,0), (int2)(0,-1)}
};

#define LAMBDA 4

// leave in float? need the precision of log2?
// mv must be in qpel
uint mv_cost(int2 mv)
{
    float2 mvc_lg2 = native_log2( convert_float2( abs( mv ) + (uint2)( 1 ) ) );
    float2 rounding = (float2)(!!mv.x, !!mv.y);
    uint2 mvc = convert_uint2(round(mvc_lg2 * 2.0f + 1.218f /*0.718f + .5f*/ + rounding));
    return LAMBDA * (mvc.x + mvc.y);
}

// no MVs to predict from
kernel void pyramid_me_stage1( read_only image2d_t fenc, read_only image2d_t fref,
                                 __global short2 *out_mv, int mb_stride )
{
    int2 mb = (int2)(get_global_id(0)*4, get_global_id(1)*16);
    int2 mv = (int2)(0);
    uint bcost = vec_sad_aligned( fenc, fref, mb, mv, 16 ) << 4;
    int i = 0;

    do
    {
        // simple diamond search
        uint costs[4] =
        {
            vec_sad_unaligned( fenc, fref, mb, mv + (int2)( 0,-1), 16 ),
            vec_sad_unaligned( fenc, fref, mb, mv + (int2)( 0, 1), 16 ),
            vec_sad_unaligned( fenc, fref, mb, mv + (int2)(-1, 0), 16 ),
            vec_sad_unaligned( fenc, fref, mb, mv + (int2)( 1, 0), 16 ),
        };
        COPY1_IF_LT( bcost, (costs[0]<<4)+1 );
        COPY1_IF_LT( bcost, (costs[1]<<4)+3 );
        COPY1_IF_LT( bcost, (costs[2]<<4)+4 );
        COPY1_IF_LT( bcost, (costs[3]<<4)+12 );
        if( !(bcost&15) )
            break;
        mv -= (int2)((bcost<<28)>>30, (bcost<<30)>>30);
        bcost &= ~15;
    } while( ++i < 16 /*me_range*/ );

    out_mv[get_global_id(0) + mb_stride * get_global_id(1)] = convert_short2(mv);
}

kernel void pyramid_me_stage2( read_only image2d_t fenc, read_only image2d_t fref,
                                 __global short2 *out_mv, int mb_stride)
{
    int2 mb = (int2)(get_global_id(0)*4, get_global_id(1)*16);
}

// Organization of group/etc: (L0: left of block 0  T1: top of block 1)
// L0 T0 L1 T1 L2 T2 L3 T3
// R0 B0 R1 B1 L2 T2 L3 T3
// L4 T4 L5 T5 L6 T6 L7 T7
// R4 B4 R5 B5 L6 T6 L7 T7
// and another 4 rows

kernel void me_pyramid(read_only image2d_t pix1, read_only image2d_t pix2,
                         __global int16_t *out)
{
    __local uint sads[8][8];
    __local short2 mvs[8][8];
    __local uint min_sad[4][4];
    __local short2 min_mv[4][4];

    int2 block = (int2)(get_global_id(0) >> 3, get_global_id(1) >> 1);
    int2 tid = (int2)(get_local_id(0), get_local_id(1));
#define TID_MOD4 (!((tid.x | tid.y) & 1))

    int2 mv = (int2)(0);

    if (TID_MOD4) {
        min_sad[tid.y>>1][tid.x>>1] = vec_sad_aligned(pix1, pix2, block, mv, 8) + mv_cost(mv);
        min_mv[tid.y>>1][tid.x>>1] = convert_short2(mv);
    }
    mv += diamond[tid.y>>2][tid.x>>2];

    sads[tid.y][tid.x] = vec_sad_unaligned(pix1, pix2, block, mv, 8);
    mvs[tid.y][tid.x] = convert_short2(mv);

    // barrier shouldn't be needed here because there are no cross-warp dependencies
    if (TID_MOD4) {
        uint min = min_sad[tid.y>>1][tid.x>>1];
        COPY2_IF_LT(min, sads[tid.y  ][tid.x  ], min_mv[tid.y>>1][tid.x>>1], mvs[tid.y  ][tid.x  ]);
        COPY2_IF_LT(min, sads[tid.y  ][tid.x+1], min_mv[tid.y>>1][tid.x>>1], mvs[tid.y  ][tid.x+1]);
        COPY2_IF_LT(min, sads[tid.y+1][tid.x  ], min_mv[tid.y>>1][tid.x>>1], mvs[tid.y+1][tid.x  ]);
        COPY2_IF_LT(min, sads[tid.y+1][tid.x+1], min_mv[tid.y>>1][tid.x>>1], mvs[tid.y+1][tid.x+1]);
        min_sad[tid.y>>1][tid.x>>1] = min;
    }

    out[block.x + block.y*256] = mv.x;//min_mv[tid.y>>1][tid.x>>1].x;
    /* This causes a compiler error with ATI's compiler.
     *      error: write to < 32 bits via pointer not
     *      allowed unless cl_khr_byte_addressable_store is enabled
     */
}

#if 0
kernel void me_full(read_only image2d_t fenc, read_only image2d_t ref,
                    global int16_t *out)
{
    int2 mb = (int2)(get_group_id(0), get_group_id(1));
    local
}
#endif
