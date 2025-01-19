import torch
import torch.nn as nn

from einops import rearrange, repeat

def hack_self_attention_to_mrsa(model, mrsa):
    """
    Hack the original self-attention module to multi-reference self-attention(MRSA) mechanism
    """
    def mrsa_forward(self, place_in_unet):
        def find_division(x, n, k):
            base_size = n // k
            extra = n % k
            if x <= extra * (base_size + 1):
                division_index = (x - 1) // (base_size + 1) + 1
            else:
                division_index = extra + ((x - 1 - extra * (base_size + 1)) // base_size + 1)
            return division_index
        
        def get_weight(index):
            if index == 1:
                return 0.77
            elif index == 2:
                return 0.88
            else:
                return 1

        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None, **kwargs):
            """
            The msra is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            timestep = kwargs.get('timestep', None)
            total_timestep = kwargs.get('total_timestep', None)
            
            index = find_division(timestep, total_timestep, 3)
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                #mask = mask * get_weight(index)
                #print("scale mask: ", mask)
                
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            sim = sim
    
            attn = sim.softmax(dim=-1)
            # the only difference
            out = mrsa(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale, **kwargs)

            return to_out(out)

        return forward
    
    def hack_attention_module(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':
                net.forward = mrsa_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = hack_attention_module(subnet, count, place_in_unet)
        return count
    
    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        
        if "down" in net_name:
            cross_att_count += hack_attention_module(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += hack_attention_module(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += hack_attention_module(net, 0, "up")
    mrsa.num_att_layers = cross_att_count