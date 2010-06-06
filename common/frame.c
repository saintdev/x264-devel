/*****************************************************************************
 * frame.c: h264 encoder library
 *****************************************************************************
 * Copyright (C) 2003-2008 x264 project
 *
 * Authors: Laurent Aimar <fenrir@via.ecp.fr>
 *          Loren Merritt <lorenm@u.washington.edu>
 *          Jason Garrett-Glaser <darkshikari@gmail.com>
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

#include "common.h"

x264_frame_t *x264_frame_new( x264_t *h, int b_fdec )
{
    x264_frame_t *frame;

    int i_mb_count = h->mb.i_mb_count;
    int i_stride, i_width, i_lines;
    int i_padv = PADV << h->param.b_interlaced;
    int luma_plane_size;
    int chroma_plane_size;
    int align = h->param.cpu&X264_CPU_CACHELINE_64 ? 64 : h->param.cpu&X264_CPU_CACHELINE_32 ? 32 : 16;

    CHECKED_MALLOCZERO( frame, sizeof(x264_frame_t) );

    /* allocate frame data (+64 for extra data for me) */
    i_width  = ALIGN( h->param.i_width, 16 );
    i_stride = ALIGN( i_width + 2*PADH, align );
    i_lines  = ALIGN( h->param.i_height, 16<<h->param.b_interlaced );

    frame->i_plane = 3;
    for( int i = 0; i < 3; i++ )
    {
        frame->i_stride[i] = ALIGN( i_stride >> !!i, align );
        frame->i_width[i] = i_width >> !!i;
        frame->i_lines[i] = i_lines >> !!i;
    }

    luma_plane_size = (frame->i_stride[0] * (frame->i_lines[0] + 2*i_padv));
    chroma_plane_size = (frame->i_stride[1] * (frame->i_lines[1] + 2*i_padv));
    for( int i = 1; i < 3; i++ )
    {
        CHECKED_MALLOC( frame->buffer[i], chroma_plane_size );
        frame->plane[i] = frame->buffer[i] + (frame->i_stride[i] * i_padv + PADH)/2;
    }

    for( int i = 0; i < h->param.i_bframe + 2; i++ )
        for( int j = 0; j < h->param.i_bframe + 2; j++ )
            CHECKED_MALLOC( frame->i_row_satds[i][j], i_lines/16 * sizeof(int) );

    frame->i_poc = -1;
    frame->i_type = X264_TYPE_AUTO;
    frame->i_qpplus1 = 0;
    frame->i_pts = -1;
    frame->i_frame = -1;
    frame->i_frame_num = -1;
    frame->i_lines_completed = -1;
    frame->b_fdec = b_fdec;
    frame->i_pic_struct = PIC_STRUCT_AUTO;
    frame->i_field_cnt = -1;
    frame->i_duration =
    frame->i_cpb_duration =
    frame->i_dpb_output_delay =
    frame->i_cpb_delay = 0;
    frame->i_coded_fields_lookahead =
    frame->i_cpb_delay_lookahead = -1;

    frame->orig = frame;

    /* all 4 luma planes allocated together, since the cacheline split code
     * requires them to be in-phase wrt cacheline alignment. */
    if( h->param.analyse.i_subpel_refine && b_fdec )
    {
        CHECKED_MALLOC( frame->buffer[0], 4*luma_plane_size );
        for( int i = 0; i < 4; i++ )
            frame->filtered[i] = frame->buffer[0] + i*luma_plane_size + frame->i_stride[0] * i_padv + PADH;
        frame->plane[0] = frame->filtered[0];
    }
    else
    {
        CHECKED_MALLOC( frame->buffer[0], luma_plane_size );
        frame->filtered[0] = frame->plane[0] = frame->buffer[0] + frame->i_stride[0] * i_padv + PADH;
    }

    frame->b_duplicate = 0;

    if( b_fdec ) /* fdec frame */
    {
        CHECKED_MALLOC( frame->mb_type, i_mb_count * sizeof(int8_t));
        CHECKED_MALLOC( frame->mb_partition, i_mb_count * sizeof(uint8_t));
        CHECKED_MALLOC( frame->mv[0], 2*16 * i_mb_count * sizeof(int16_t) );
        CHECKED_MALLOC( frame->mv16x16, 2*i_mb_count * sizeof(int16_t) );
        CHECKED_MALLOC( frame->ref[0], 4 * i_mb_count * sizeof(int8_t) );
        if( h->param.i_bframe )
        {
            CHECKED_MALLOC( frame->mv[1], 2*16 * i_mb_count * sizeof(int16_t) );
            CHECKED_MALLOC( frame->ref[1], 4 * i_mb_count * sizeof(int8_t) );
        }
        else
        {
            frame->mv[1]  = NULL;
            frame->ref[1] = NULL;
        }
        CHECKED_MALLOC( frame->i_row_bits, i_lines/16 * sizeof(int) );
        CHECKED_MALLOC( frame->f_row_qp, i_lines/16 * sizeof(float) );
        if( h->param.analyse.i_me_method >= X264_ME_ESA )
        {
            CHECKED_MALLOC( frame->buffer[3],
                            frame->i_stride[0] * (frame->i_lines[0] + 2*i_padv) * sizeof(uint16_t) << h->frames.b_have_sub8x8_esa );
            frame->integral = (uint16_t*)frame->buffer[3] + frame->i_stride[0] * i_padv + PADH;
        }
    }
    else /* fenc frame */
    {
        if( h->frames.b_have_lowres )
        {
            frame->i_width_lowres = frame->i_width[0]/2;
            frame->i_stride_lowres = ALIGN( frame->i_width_lowres + 2*PADH, align );
            frame->i_lines_lowres = frame->i_lines[0]/2;

            luma_plane_size = frame->i_stride_lowres * (frame->i_lines[0]/2 + 2*PADV);

            CHECKED_MALLOC( frame->buffer_lowres[0], 4 * luma_plane_size );
            for( int i = 0; i < 4; i++ )
                frame->lowres[i] = frame->buffer_lowres[0] + (frame->i_stride_lowres * PADV + PADH) + i * luma_plane_size;

            for( int j = 0; j <= !!h->param.i_bframe; j++ )
                for( int i = 0; i <= h->param.i_bframe; i++ )
                {
                    CHECKED_MALLOCZERO( frame->lowres_mvs[j][i], 2*h->mb.i_mb_count*sizeof(int16_t) );
                    CHECKED_MALLOC( frame->lowres_mv_costs[j][i], h->mb.i_mb_count*sizeof(int) );
                }
            CHECKED_MALLOC( frame->i_propagate_cost, (i_mb_count+3) * sizeof(uint16_t) );
            for( int j = 0; j <= h->param.i_bframe+1; j++ )
                for( int i = 0; i <= h->param.i_bframe+1; i++ )
                    CHECKED_MALLOC( frame->lowres_costs[j][i], (i_mb_count+3) * sizeof(uint16_t) );
            frame->i_intra_cost = frame->lowres_costs[0][0];
            memset( frame->i_intra_cost, -1, (i_mb_count+3) * sizeof(uint16_t) );
        }
        if( h->param.rc.i_aq_mode )
        {
            CHECKED_MALLOC( frame->f_qp_offset, h->mb.i_mb_count * sizeof(float) );
            CHECKED_MALLOC( frame->f_qp_offset_aq, h->mb.i_mb_count * sizeof(float) );
            if( h->frames.b_have_lowres )
                /* shouldn't really be initialized, just silences a valgrind false-positive in x264_mbtree_propagate_cost_sse2 */
                CHECKED_MALLOCZERO( frame->i_inv_qscale_factor, (h->mb.i_mb_count+3) * sizeof(uint16_t) );
        }
    }

    if( x264_pthread_mutex_init( &frame->mutex, NULL ) )
        goto fail;
    if( x264_pthread_cond_init( &frame->cv, NULL ) )
        goto fail;

    return frame;

fail:
    x264_free( frame );
    return NULL;
}

void x264_frame_delete( x264_frame_t *frame )
{
    /* Duplicate frames are blank copies of real frames (including pointers),
     * so freeing those pointers would cause a double free later. */
    if( !frame->b_duplicate )
    {
        for( int i = 0; i < 4; i++ )
            x264_free( frame->buffer[i] );
        for( int i = 0; i < 4; i++ )
            x264_free( frame->buffer_lowres[i] );
        for( int i = 0; i < X264_BFRAME_MAX+2; i++ )
            for( int j = 0; j < X264_BFRAME_MAX+2; j++ )
                x264_free( frame->i_row_satds[i][j] );
        for( int j = 0; j < 2; j++ )
            for( int i = 0; i <= X264_BFRAME_MAX; i++ )
            {
                x264_free( frame->lowres_mvs[j][i] );
                x264_free( frame->lowres_mv_costs[j][i] );
            }
        x264_free( frame->i_propagate_cost );
        for( int j = 0; j <= X264_BFRAME_MAX+1; j++ )
            for( int i = 0; i <= X264_BFRAME_MAX+1; i++ )
                x264_free( frame->lowres_costs[j][i] );
        x264_free( frame->f_qp_offset );
        x264_free( frame->f_qp_offset_aq );
        x264_free( frame->i_inv_qscale_factor );
        x264_free( frame->i_row_bits );
        x264_free( frame->f_row_qp );
        x264_free( frame->mb_type );
        x264_free( frame->mb_partition );
        x264_free( frame->mv[0] );
        x264_free( frame->mv[1] );
        x264_free( frame->mv16x16 );
        x264_free( frame->ref[0] );
        x264_free( frame->ref[1] );
        x264_pthread_mutex_destroy( &frame->mutex );
        x264_pthread_cond_destroy( &frame->cv );
    }
    x264_free( frame );
}

int x264_frame_copy_picture( x264_t *h, x264_frame_t *dst, x264_picture_t *src )
{
    int i_csp = src->img.i_csp & X264_CSP_MASK;
    if( i_csp != X264_CSP_I420 && i_csp != X264_CSP_YV12 )
    {
        x264_log( h, X264_LOG_ERROR, "Invalid input colorspace\n" );
        return -1;
    }

    dst->i_type     = src->i_type;
    dst->i_qpplus1  = src->i_qpplus1;
    dst->i_pts      = dst->i_reordered_pts = src->i_pts;
    dst->param      = src->param;
    dst->i_pic_struct = src->i_pic_struct;

    for( int i = 0; i < 3; i++ )
    {
        int s = (i_csp == X264_CSP_YV12 && i) ? i^3 : i;
        uint8_t *plane = src->img.plane[s];
        int stride = src->img.i_stride[s];
        int width = h->param.i_width >> !!i;
        int height = h->param.i_height >> !!i;
        if( src->img.i_csp & X264_CSP_VFLIP )
        {
            plane += (height-1)*stride;
            stride = -stride;
        }
        if( width > abs(stride) )
        {
            x264_log( h, X264_LOG_ERROR, "Input picture width is greater than stride\n" );
            return -1;
        }
        h->mc.plane_copy( dst->plane[i], dst->i_stride[i], plane, stride, width, height );
    }
    return 0;
}



static void plane_expand_border( uint8_t *pix, int i_stride, int i_width, int i_height, int i_padh, int i_padv, int b_pad_top, int b_pad_bottom )
{
#define PPIXEL(x, y) ( pix + (x) + (y)*i_stride )
    for( int y = 0; y < i_height; y++ )
    {
        /* left band */
        memset( PPIXEL(-i_padh, y), PPIXEL(0, y)[0], i_padh );
        /* right band */
        memset( PPIXEL(i_width, y), PPIXEL(i_width-1, y)[0], i_padh );
    }
    /* upper band */
    if( b_pad_top )
        for( int y = 0; y < i_padv; y++ )
            memcpy( PPIXEL(-i_padh, -y-1), PPIXEL(-i_padh, 0), i_width+2*i_padh );
    /* lower band */
    if( b_pad_bottom )
        for( int y = 0; y < i_padv; y++ )
            memcpy( PPIXEL(-i_padh, i_height+y), PPIXEL(-i_padh, i_height-1), i_width+2*i_padh );
#undef PPIXEL
}

void x264_frame_expand_border( x264_t *h, x264_frame_t *frame, int mb_y, int b_end )
{
    int b_start = !mb_y;
    if( mb_y & h->sh.b_mbaff )
        return;
    for( int i = 0; i < frame->i_plane; i++ )
    {
        int stride = frame->i_stride[i];
        int width = 16*h->sps->i_mb_width >> !!i;
        int height = (b_end ? 16*(h->sps->i_mb_height - mb_y) >> h->sh.b_mbaff : 16) >> !!i;
        int padh = PADH >> !!i;
        int padv = PADV >> !!i;
        // buffer: 2 chroma, 3 luma (rounded to 4) because deblocking goes beyond the top of the mb
        uint8_t *pix = frame->plane[i] + X264_MAX(0, (16*mb_y-4)*stride >> !!i);
        if( b_end && !b_start )
            height += 4 >> (!!i + h->sh.b_mbaff);
        if( h->sh.b_mbaff )
        {
            plane_expand_border( pix, stride*2, width, height, padh, padv, b_start, b_end );
            plane_expand_border( pix+stride, stride*2, width, height, padh, padv, b_start, b_end );
        }
        else
        {
            plane_expand_border( pix, stride, width, height, padh, padv, b_start, b_end );
        }
    }
}

void x264_frame_expand_border_filtered( x264_t *h, x264_frame_t *frame, int mb_y, int b_end )
{
    /* during filtering, 8 extra pixels were filtered on each edge,
     * but up to 3 of the horizontal ones may be wrong.
       we want to expand border from the last filtered pixel */
    int b_start = !mb_y;
    int stride = frame->i_stride[0];
    int width = 16*h->sps->i_mb_width + 8;
    int height = b_end ? (16*(h->sps->i_mb_height - mb_y) >> h->sh.b_mbaff) + 16 : 16;
    int padh = PADH - 4;
    int padv = PADV - 8;
    for( int i = 1; i < 4; i++ )
    {
        // buffer: 8 luma, to match the hpel filter
        uint8_t *pix = frame->filtered[i] + (16*mb_y - (8 << h->sh.b_mbaff)) * stride - 4;
        if( h->sh.b_mbaff )
        {
            plane_expand_border( pix, stride*2, width, height, padh, padv, b_start, b_end );
            plane_expand_border( pix+stride, stride*2, width, height, padh, padv, b_start, b_end );
        }
        else
            plane_expand_border( pix, stride, width, height, padh, padv, b_start, b_end );
    }
}

void x264_frame_expand_border_lowres( x264_frame_t *frame )
{
    for( int i = 0; i < 4; i++ )
        plane_expand_border( frame->lowres[i], frame->i_stride_lowres, frame->i_width_lowres, frame->i_lines_lowres, PADH, PADV, 1, 1 );
}

void x264_frame_expand_border_mod16( x264_t *h, x264_frame_t *frame )
{
    for( int i = 0; i < frame->i_plane; i++ )
    {
        int i_subsample = i ? 1 : 0;
        int i_width = h->param.i_width >> i_subsample;
        int i_height = h->param.i_height >> i_subsample;
        int i_padx = (h->sps->i_mb_width * 16 - h->param.i_width) >> i_subsample;
        int i_pady = (h->sps->i_mb_height * 16 - h->param.i_height) >> i_subsample;

        if( i_padx )
        {
            for( int y = 0; y < i_height; y++ )
                memset( &frame->plane[i][y*frame->i_stride[i] + i_width],
                         frame->plane[i][y*frame->i_stride[i] + i_width - 1],
                         i_padx );
        }
        if( i_pady )
        {
            for( int y = i_height; y < i_height + i_pady; y++ )
                memcpy( &frame->plane[i][y*frame->i_stride[i]],
                        &frame->plane[i][(i_height-(~y&h->param.b_interlaced)-1)*frame->i_stride[i]],
                        i_width + i_padx );
        }
    }
}

/* threading */
void x264_frame_cond_broadcast( x264_frame_t *frame, int i_lines_completed )
{
    x264_pthread_mutex_lock( &frame->mutex );
    frame->i_lines_completed = i_lines_completed;
    x264_pthread_cond_broadcast( &frame->cv );
    x264_pthread_mutex_unlock( &frame->mutex );
}

void x264_frame_cond_wait( x264_frame_t *frame, int i_lines_completed )
{
    x264_pthread_mutex_lock( &frame->mutex );
    while( frame->i_lines_completed < i_lines_completed )
        x264_pthread_cond_wait( &frame->cv, &frame->mutex );
    x264_pthread_mutex_unlock( &frame->mutex );
}

/* list operators */

void x264_frame_push( x264_frame_t **list, x264_frame_t *frame )
{
    int i = 0;
    while( list[i] ) i++;
    list[i] = frame;
}

x264_frame_t *x264_frame_pop( x264_frame_t **list )
{
    x264_frame_t *frame;
    int i = 0;
    assert( list[0] );
    while( list[i+1] ) i++;
    frame = list[i];
    list[i] = NULL;
    return frame;
}

void x264_frame_unshift( x264_frame_t **list, x264_frame_t *frame )
{
    int i = 0;
    while( list[i] ) i++;
    while( i-- )
        list[i+1] = list[i];
    list[0] = frame;
}

x264_frame_t *x264_frame_shift( x264_frame_t **list )
{
    x264_frame_t *frame = list[0];
    int i;
    for( i = 0; list[i]; i++ )
        list[i] = list[i+1];
    assert(frame);
    return frame;
}

void x264_frame_push_unused( x264_t *h, x264_frame_t *frame )
{
    assert( frame->i_reference_count > 0 );
    frame->i_reference_count--;
    if( frame->i_reference_count == 0 )
        x264_frame_push( h->frames.unused[frame->b_fdec], frame );
}

x264_frame_t *x264_frame_pop_unused( x264_t *h, int b_fdec )
{
    x264_frame_t *frame;
    if( h->frames.unused[b_fdec][0] )
        frame = x264_frame_pop( h->frames.unused[b_fdec] );
    else
        frame = x264_frame_new( h, b_fdec );
    if( !frame )
        return NULL;
    frame->b_last_minigop_bframe = 0;
    frame->i_reference_count = 1;
    frame->b_intra_calculated = 0;
    frame->b_scenecut = 1;
    frame->b_keyframe = 0;

    memset( frame->weight, 0, sizeof(frame->weight) );
    memset( frame->f_weighted_cost_delta, 0, sizeof(frame->f_weighted_cost_delta) );

    return frame;
}

void x264_frame_push_blank_unused( x264_t *h, x264_frame_t *frame )
{
    assert( frame->i_reference_count > 0 );
    frame->i_reference_count--;
    if( frame->i_reference_count == 0 )
        x264_frame_push( h->frames.blank_unused, frame );
}

x264_frame_t *x264_frame_pop_blank_unused( x264_t *h )
{
    x264_frame_t *frame;
    if( h->frames.blank_unused[0] )
        frame = x264_frame_pop( h->frames.blank_unused );
    else
        frame = x264_malloc( sizeof(x264_frame_t) );
    if( !frame )
        return NULL;
    frame->b_duplicate = 1;
    frame->i_reference_count = 1;
    return frame;
}

void x264_frame_sort( x264_frame_t **list, int b_dts )
{
    int b_ok;
    do {
        b_ok = 1;
        for( int i = 0; list[i+1]; i++ )
        {
            int dtype = list[i]->i_type - list[i+1]->i_type;
            int dtime = list[i]->i_frame - list[i+1]->i_frame;
            int swap = b_dts ? dtype > 0 || ( dtype == 0 && dtime > 0 )
                             : dtime > 0;
            if( swap )
            {
                XCHG( x264_frame_t*, list[i], list[i+1] );
                b_ok = 0;
            }
        }
    } while( !b_ok );
}

void x264_weight_scale_plane( x264_t *h, uint8_t *dst, int i_dst_stride, uint8_t *src, int i_src_stride,
                         int i_width, int i_height, x264_weight_t *w )
{
    /* Weight horizontal strips of height 16. This was found to be the optimal height
     * in terms of the cache loads. */
    while( i_height > 0 )
    {
        for( int x = 0; x < i_width; x += 16 )
            w->weightfn[16>>2]( dst+x, i_dst_stride, src+x, i_src_stride, w, X264_MIN( i_height, 16 ) );
        i_height -= 16;
        dst += 16 * i_dst_stride;
        src += 16 * i_src_stride;
    }
}

void x264_frame_delete_list( x264_frame_t **list )
{
    int i = 0;
    if( !list )
        return;
    while( list[i] )
        x264_frame_delete( list[i++] );
    x264_free( list );
}

int x264_synch_frame_list_init( x264_synch_frame_list_t *slist, int max_size )
{
    if( max_size < 0 )
        return -1;
    slist->i_max_size = max_size;
    slist->i_size = 0;
    CHECKED_MALLOCZERO( slist->list, (max_size+1) * sizeof(x264_frame_t*) );
    if( x264_pthread_mutex_init( &slist->mutex, NULL ) ||
        x264_pthread_cond_init( &slist->cv_fill, NULL ) ||
        x264_pthread_cond_init( &slist->cv_empty, NULL ) )
        return -1;
    return 0;
fail:
    return -1;
}

void x264_synch_frame_list_delete( x264_synch_frame_list_t *slist )
{
    x264_pthread_mutex_destroy( &slist->mutex );
    x264_pthread_cond_destroy( &slist->cv_fill );
    x264_pthread_cond_destroy( &slist->cv_empty );
    x264_frame_delete_list( slist->list );
}

void x264_synch_frame_list_push( x264_synch_frame_list_t *slist, x264_frame_t *frame )
{
    x264_pthread_mutex_lock( &slist->mutex );
    while( slist->i_size == slist->i_max_size )
        x264_pthread_cond_wait( &slist->cv_empty, &slist->mutex );
    slist->list[ slist->i_size++ ] = frame;
    x264_pthread_mutex_unlock( &slist->mutex );
    x264_pthread_cond_broadcast( &slist->cv_fill );
}

void x264_synch_frame_list_shift( x264_synch_frame_list_t *dst, x264_synch_frame_list_t *src, int count )
{
    int i = count;
    while( i-- )
    {
        assert( dst->i_size < dst->i_max_size );
        assert( src->i_size );
        dst->list[ dst->i_size++ ] = x264_frame_shift( src->list );
        src->i_size--;
    }
    if( count )
    {
        x264_pthread_cond_broadcast( &dst->cv_fill );
        x264_pthread_cond_broadcast( &src->cv_empty );
    }
}
