import numpy as np
import torch
from typing import Optional, Union, Tuple, Dict
from PIL import Image


def save_images(images,dest, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    pil_img = Image.fromarray(images[-1])
    pil_img.save(dest)
    # display(pil_img)


def save_image(images,dest, num_rows=1, offset_ratio=0.02):
    print(images.shape)
    pil_img = Image.fromarray(images[0])
    pil_img.save(dest)

def register_attention_control_all(model, controller, self_attn_list=[]):
    class AttnProcessor():
        def __init__(self,place_in_unet):
            self.place_in_unet = place_in_unet

        def __call__(self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,):
            # The `Attention` class can call different attention processors / attention functions
            residual = hidden_states

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            h = attn.heads
            is_cross = encoder_hidden_states is not None 
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            q = attn.to_q(hidden_states)
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)
            q = attn.head_to_batch_dim(q)
            k = attn.head_to_batch_dim(k)
            v = attn.head_to_batch_dim(v)
            # import pdb; pdb.set_trace()

            if not is_cross:
                qu_list = []
                qc_list = []
                ku_list = []
                kc_list = []
                vu_list = []
                vc_list = []
                qu, qc = q.chunk(2)
                ku, kc = k.chunk(2)
                vu, vc = v.chunk(2)
                # import pdb; pdb.set_trace()
                for idx, ctrl in enumerate(controller):
                    
                    if idx in self_attn_list:
                        if idx==0:
                            qu_list.append(qu[:3*attn.heads])
                            qc_list.append(qc[:3*attn.heads])
                            ku_list.append(ku[:3*attn.heads])
                            kc_list.append(kc[:3*attn.heads])
                            vu_list.append(vu[:3*attn.heads])
                            vc_list.append(vc[:3*attn.heads])
                        else:
                            range2 = range((idx*2+1)*attn.heads, (idx*2+3)*attn.heads)
                            qu_list.append(qu[range2])
                            qc_list.append(qc[range2])
                            ku_list.append(ku[range2])
                            kc_list.append(kc[range2])
                            vu_list.append(vu[range2])
                            vc_list.append(vc[range2])
                    elif idx == 0:
                        iqu, iqc, iku, ikc, ivu, ivc = ctrl.self_attn_forward(qu[:3*attn.heads], qc[:3*attn.heads], ku[:3*attn.heads], kc[:3*attn.heads], vu[:3*attn.heads], vc[:3*attn.heads], attn.heads)
                        qu_list.append(iqu)
                        qc_list.append(iqc)
                        ku_list.append(iku)
                        kc_list.append(ikc)
                        vu_list.append(ivu)
                        vc_list.append(ivc)
                    else:
                        range1 = range((idx*2-1)*attn.heads, idx*2*attn.heads)
                        range2 = range((idx*2+1)*attn.heads, (idx*2+3)*attn.heads)
                        iqu, iqc, iku, ikc, ivu, ivc = ctrl.self_attn_forward(
                            torch.cat([qu[range1], qu[range2]]), 
                            torch.cat([qc[range1], qc[range2]]),
                            torch.cat([ku[range1], ku[range2]]),
                            torch.cat([kc[range1], kc[range2]]),
                            torch.cat([vu[range1], vu[range2]]),
                            torch.cat([vc[range1], vc[range2]]), attn.heads)
                
                        qu_list.append(iqu[attn.heads:])
                        qc_list.append(iqc[attn.heads:])
                        ku_list.append(iku[attn.heads:])
                        kc_list.append(ikc[attn.heads:])
                        vu_list.append(ivu[attn.heads:])
                        vc_list.append(ivc[attn.heads:])
                q = torch.cat(qu_list + qc_list)  
                k = torch.cat(ku_list + kc_list)
                v = torch.cat(vu_list + vc_list)

              # import pdb; pdb.set_trace()
            try:
                attention_probs = attn.get_attention_scores(q, k, attention_mask)
            except:
                import pdb; pdb.set_trace()
            if is_cross:                
                attention_probs_u, attention_probs_c = attention_probs.chunk(2)
                attention_probs_c_list = []
                attention_probs_u_list = []
                # attention_probs_u = [attention_probs_u]
                # import pdb; pdb.set_trace()
                for idx in range(len(controller)):
                    if idx == 0:
                        # import pdb; pdb.set_trace()
                        attention_prob = torch.cat([attention_probs_u[:3*attn.heads], attention_probs_c[:3*attn.heads]])
                        attention_prob  = controller[idx](attention_prob , is_cross, self.place_in_unet) 
                        attention_probs_u_list.append(attention_prob[:3*attn.heads])
                        attention_probs_c_list.append(attention_prob[3*attn.heads:])   
                    else:
                        attention_prob = torch.cat([attention_probs_u[(idx*2-1)*attn.heads:idx*2*attn.heads], attention_probs_u[(idx*2+1)*attn.heads:(idx*2+3)*attn.heads], 
                                                attention_probs_c[(idx*2-1)*attn.heads:idx*2*attn.heads], attention_probs_c[(idx*2+1)*attn.heads:(idx*2+3)*attn.heads]])
                        attention_prob  = controller[idx](attention_prob , is_cross, self.place_in_unet) 
                        attention_probs_u_list.append(attention_prob[attn.heads:3*attn.heads] )
                        attention_probs_c_list.append(attention_prob[4*attn.heads:])

                attention_probs = torch.cat(attention_probs_u_list + attention_probs_c_list)
            hidden_states = torch.bmm(attention_probs, v)
            # try:
            hidden_states = attn.batch_to_head_dim(hidden_states)
            # except:
                # import pdb; pdb.set_trace()

            # linear proj   
            hidden_states = attn.to_out[0](hidden_states, scale=scale)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor
            # import pdb; pdb.set_trace()
            return hidden_states


    def register_recr(net_, count, place_in_unet):
        for idx, m in enumerate(net_.modules()):
            if m.__class__.__name__ == "Attention":
                count+=1
                m.processor = AttnProcessor( place_in_unet)
        return count

    if not isinstance(controller, list):
        controller = [controller]
    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    
    if isinstance(controller, list):
        for c in controller:
            c.num_att_layers = cross_att_count
    else:
        controller.num_att_layers = cross_att_count

def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)

def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int, word_inds: Optional[torch.Tensor]=None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha

def get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words) # time, batch, heads, pixels, words
    return alpha_time_words
