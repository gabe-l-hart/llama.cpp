from __future__ import annotations

import re
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf, logger

from .llama import LlamaModel
from .mamba import Mamba2Model


@ModelBase.register("GraniteForCausalLM")
class GraniteModel(LlamaModel):
    """Conversion for IBM's GraniteForCausalLM"""
    model_arch = gguf.MODEL_ARCH.GRANITE

    def set_gguf_parameters(self):
        """Granite uses standard llama parameters with the following differences:

        - No head_dim support
        - New multiplier params:
            - attention_scale
            - embedding_scale
            - residual_scale
        - logits_scaling
        """
        if head_dim := self.hparams.pop("head_dim", None):
            logger.warning("Ignoring head_dim (%s) from config for Granite", head_dim)
        super().set_gguf_parameters()
        # NOTE: Convert _multiplier params to _scale params for naming
        #   consistency
        if attention_scale := self.hparams.get("attention_multiplier"):
            self.gguf_writer.add_attention_scale(attention_scale)
            logger.info("gguf: (granite) attention_scale = %s", attention_scale)
        if embedding_scale := self.hparams.get("embedding_multiplier"):
            self.gguf_writer.add_embedding_scale(embedding_scale)
            logger.info("gguf: (granite) embedding_scale = %s", embedding_scale)
        if residual_scale := self.hparams.get("residual_multiplier"):
            self.gguf_writer.add_residual_scale(residual_scale)
            logger.info("gguf: (granite) residual_scale = %s", residual_scale)
        if logits_scale := self.hparams.get("logits_scaling"):
            self.gguf_writer.add_logit_scale(logits_scale)
            logger.info("gguf: (granite) logits_scale = %s", logits_scale)

        # If being used as the base for Granite4 Vision, add deepstack_layer_arr
        if self.hparams.get("spatial_target_layers") or self.hparams.get("deepstack_layer_map"):
            normalized_projector_map = Granite4VisionMmprojModel.get_normalized_projector_map(self.hparams)
            deepstack_mapping_arr = [-1 for _ in range(self.block_count)] # Populate with -1 sentinels
            for proj_idx, (_, llm_layer, _, _) in enumerate(normalized_projector_map):
                # Skip the first projector which is handled as the base embedding
                # stream like normal
                if proj_idx == 0:
                    continue
                deepstack_mapping_arr[llm_layer] = proj_idx
            self.gguf_writer.add_deepstack_mapping(deepstack_mapping_arr)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        # Skip multimodal tensors
        if (
            name.startswith(("encoder."))
            or "image_" in name
            or "layerwise_projectors" in name
            or "spatial_projectors" in name
        ):
            return
        return super().filter_tensors(item)


@ModelBase.register("GraniteMoeForCausalLM", "GraniteMoeSharedForCausalLM")
class GraniteMoeModel(GraniteModel):
    """Conversion for IBM's GraniteMoeForCausalLM"""
    model_arch = gguf.MODEL_ARCH.GRANITE_MOE

    def set_gguf_parameters(self):
        """GraniteMoeShared uses GraniteMoe parameters plus the following:
        - shared_intermediate_size
        """
        super().set_gguf_parameters()
        if shared_feed_forward_length := self.hparams.get("shared_intermediate_size"):
            self.gguf_writer.add_expert_shared_feed_forward_length(shared_feed_forward_length)
            logger.info("gguf: (granitemoeshared) shared_feed_forward_length = %s", shared_feed_forward_length)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """In modeling_granitemoe, the JetMoe implementation of parallel experts
        is used. This essentially merges w1 and w3 into a single tensor with 2x
        the hidden size that is then split during forward. To keep compatibility
        with existing mixtral support, we pull them apart here.
        """

        if name.endswith("block_sparse_moe.input_linear.weight"):
            ffn_dim = self.hparams["intermediate_size"]
            assert data_torch.shape[-2] == 2 * ffn_dim, "Merged FFN tensor size must be 2 * intermediate_size"
            gate, up = data_torch.split(ffn_dim, dim=-2)
            yield from ModelBase.modify_tensors(self, gate, self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_EXP, bid), bid)
            yield from ModelBase.modify_tensors(self, up, self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP_EXP, bid), bid)
            return

        has_experts = bool(self.hparams.get('num_local_experts'))

        if name.endswith("shared_mlp.input_linear.weight"):
            ffn_dim = self.hparams["shared_intermediate_size"]
            assert data_torch.shape[-2] == 2 * ffn_dim, "Merged FFN tensor size must be 2 * shared_intermediate_size"
            gate, up = data_torch.split(ffn_dim, dim=-2)
            if has_experts:
                yield from ModelBase.modify_tensors(self, gate,self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_SHEXP, bid), bid)
                yield from ModelBase.modify_tensors(self, up, self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP_SHEXP, bid), bid)
                return
            yield from ModelBase.modify_tensors(self, gate, self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE, bid), bid)
            yield from ModelBase.modify_tensors(self, up, self.format_tensor_name(gguf.MODEL_TENSOR.FFN_UP, bid), bid)
            return

        if not has_experts and name.endswith("shared_mlp.output_linear.weight"):
            yield from ModelBase.modify_tensors(self, data_torch, self.format_tensor_name(gguf.MODEL_TENSOR.FFN_DOWN, bid), bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("GraniteMoeHybridForCausalLM", "BambaForCausalLM")
class GraniteHybridModel(Mamba2Model, GraniteMoeModel):
    """GraniteHybrid is a hybrid SSM + Attention model that uses Mamba2 SSM
    layers and optionally uses MoE w/ a shared expert"""
    model_arch = gguf.MODEL_ARCH.GRANITE_HYBRID
    undo_permute = True

    def __init__(self, *args, **kwargs):

        # Hybrid mamba models use a prefix for the mamba-specific params.
        # TODO: Extend this if the prefix(es) need to be configurable
        self.hparam_prefixes = ["mamba"]

        super().__init__(*args, **kwargs)

        # Lists of which layers use ssm vs attention
        self._attn_layers = self.get_attn_layers()
        self._ssm_layers = [
            i for i in range(self.block_count)
            if i not in self._attn_layers
        ]

        # There are some models in this family that are non-hybrid, but keep the
        # same parent class by setting all layers to "attention." If this is the
        # case, the model architecture needs to be updated to a standard
        # "granite" or "granitemoe" model
        if not self._ssm_layers:
            has_experts = self.find_hparam(["num_experts_per_tok", "num_experts_per_token"], optional=True)
            new_arch = (
                gguf.MODEL_ARCH.GRANITE_MOE
                if has_experts else
                gguf.MODEL_ARCH.GRANITE
            )
            self.model_arch = new_arch
            self.gguf_writer.arch = gguf.MODEL_ARCH_NAMES[new_arch]
            self.gguf_writer.add_architecture()

        # n_group and d_inner are used during reshape_tensors for mamba2
        # NOTE: Explicitly include hparam prefix prefix for d_model to
        #   disambiguate with top-level head_dim
        # NOTE 2: If needed for future models, this can be isolated in a method
        #   to separate the prefix setting and the keys used
        self.d_model = self.find_hparam([f"{self.hparam_prefixes[0]}_head_dim", "hidden_size", "d_model"])
        self.n_group = self.find_hparam(["n_groups", "num_groups"])
        self.d_inner = self.find_hparam(["expand", "num_heads"]) * self.d_model

    def get_attn_layers(self):
        # Explicit list of layer type names
        if layer_types := self.hparams.get("layer_types"):
            return [
                i for i, typ in enumerate(layer_types)
                if typ == "attention"
            ]

        # Layer types indicated by index or period
        attn_layers = self.hparams.get("attn_layer_indices", [])
        if not attn_layers:
            attn_period = self.hparams.get("attn_layer_period")
            assert attn_period, "Didn't find attn_layer_indices or attn_layer_period"
            attn_offset = self.hparams.get("attn_layer_offset")
            assert attn_offset is not None, "No attention layer offset set with attn_layer_period"
            attn_layers = [
                i for i in range(self.block_count)
                if i % attn_period == attn_offset
            ]
        return attn_layers

    def find_hparam(self, keys: Iterable[str], *args, **kwargs) -> Any:
        prefixed = []
        for pfx in self.hparam_prefixes:
            prefixed.extend(
                "_".join([pfx, k])
                for k in keys
            )
        keys = list(keys) + prefixed
        return Mamba2Model.find_hparam(self, keys, *args, **kwargs)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if (
            name.endswith("block_sparse_moe.input_linear.weight")
            or "shared_mlp" in name
        ):
            yield from GraniteMoeModel.modify_tensors(self, data_torch, name, bid)
            return

        # Determine whether this is a mamba layer or an attention layer
        if bid in self._ssm_layers:
            yield from Mamba2Model.modify_tensors(self, data_torch, name, bid)
            return
        elif bid in self._attn_layers:
            yield from GraniteMoeModel.modify_tensors(self, data_torch, name, bid)
            return
        yield from ModelBase.modify_tensors(self, data_torch, name, bid)

    def set_gguf_parameters(self):
        """This method merges params from both parents and some that are
        specific to this model. The result is some duplication of how the params
        get set. The following warnings are expected during conversion:

        WARNING:Duplicated key name 'granitehybrid.attention.head_count_kv'
        WARNING:Duplicated key name 'granitehybrid.context_length'
        """
        GraniteMoeModel.set_gguf_parameters(self)

        ## Mamba mixer params ##
        self.gguf_writer.add_ssm_conv_kernel(self.find_hparam(["conv_kernel", "d_conv"]))
        self.gguf_writer.add_ssm_state_size(self.find_hparam(["state_size", "d_state", "state_dim", "ssm_state_size"]))
        self.gguf_writer.add_ssm_group_count(self.n_group)
        self.gguf_writer.add_ssm_inner_size(self.d_inner)
        # NOTE: The mamba_dt_rank is _not_ the right field for how this is used
        #   in llama.cpp
        self.gguf_writer.add_ssm_time_step_rank(self.find_hparam(["n_heads", "num_heads"]))

        ## Attention params ##
        head_count_kv = self.find_hparam(["num_key_value_heads", "n_head_kv"])
        head_count_kv_vec = [
            head_count_kv if i in self._attn_layers else 0 for i in range(self.block_count)
        ]
        if rope_dim := self.hparams.get("attn_rotary_emb"):
            self.gguf_writer.add_rope_dimension_count(rope_dim)
        self.gguf_writer.add_head_count_kv(head_count_kv_vec)

        ## If Bamba or non-hybrid, use rope, otherwise don't
        use_rope = (
            "BambaForCausalLM" in self.hparams["architectures"]
            or not self._ssm_layers
        )
        self.gguf_writer.add_rope_scaling_finetuned(use_rope)
        if not use_rope:
            self.gguf_writer.add_context_length(2**20)

        ## Validation ##
        d_head = self.find_hparam(["d_head"], optional=True) or 64
        assert self.hparams.get("hidden_act") in [None, "silu"], "Only SILU activation supported"
        assert self.d_inner % d_head == 0, f"SSM inner size {self.d_inner} not a multiple of head dim {d_head}"

    def set_vocab(self):
        # For models with no ssm layers, don't pad for mamba2
        self.hparams["pad_vocab_size_multiple"] = 8 if self._ssm_layers else 1
        Mamba2Model.set_vocab(self)


@ModelBase.register("GraniteSpeechForConditionalGeneration")
class GraniteSpeechMmprojModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = True

    _batch_norm_tensors: list[dict[str, Tensor]] | None = None

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config.get("encoder_config")

    def set_gguf_parameters(self):
        assert self.hparams_audio is not None
        a = self.hparams_audio
        a["hidden_size"] = a["hidden_dim"]
        a["intermediate_size"] = a["hidden_dim"] * a["feedforward_mult"]
        a["num_attention_heads"] = a["num_heads"]
        a["num_hidden_layers"] = a["num_layers"]

        super().set_gguf_parameters()

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.GRANITE_SPEECH)
        self.gguf_writer.add_audio_num_mel_bins(a["input_dim"])
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-5)
        self.gguf_writer.add_audio_chunk_size(a["context_size"])
        self.gguf_writer.add_audio_conv_kernel_size(a["conv_kernel_size"])
        self.gguf_writer.add_audio_max_pos_emb(a["max_pos_emb"])

        p = self.global_config
        self.gguf_writer.add_audio_projector_window_size(p["window_size"])
        self.gguf_writer.add_audio_projector_downsample_rate(p["downsample_rate"])
        self.gguf_writer.add_audio_projector_head_count(p["projector_config"]["num_attention_heads"])

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if "encoder" in name or "projector" in name:
            if ".conv" in name and ".weight" in name:
                return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if "attention_dists" in name or "num_batches_tracked" in name:
            return None
        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # fold running_mean, running_var and eps into weight and bias for batch_norm
        if "batch_norm" in name and "encoder.layers." in name:
            if self._batch_norm_tensors is None:
                self._batch_norm_tensors = [{} for _ in range(self.block_count)]
            assert bid is not None
            self._batch_norm_tensors[bid][name] = data_torch
            if len(self._batch_norm_tensors[bid]) < 4:
                return
            prefix = f"encoder.layers.{bid}.conv.batch_norm"
            weight = self._batch_norm_tensors[bid][f"{prefix}.weight"]
            bias = self._batch_norm_tensors[bid][f"{prefix}.bias"]
            running_mean = self._batch_norm_tensors[bid][f"{prefix}.running_mean"]
            running_var = self._batch_norm_tensors[bid][f"{prefix}.running_var"]
            eps = 1e-5
            a = weight / torch.sqrt(running_var + eps)
            b = bias - running_mean * a
            yield from super().modify_tensors(a, f"encoder.layers.{bid}.conv.batch_norm.weight", bid)
            yield from super().modify_tensors(b, f"encoder.layers.{bid}.conv.batch_norm.bias", bid)
            return

        if ".attn.to_kv.weight" in name:
            k_weight, v_weight = data_torch.chunk(2, dim=0)
            yield from super().modify_tensors(k_weight, name.replace("to_kv", "to_k"), bid)
            yield from super().modify_tensors(v_weight, name.replace("to_kv", "to_v"), bid)
            return

        if ("up_conv" in name or "down_conv" in name) and name.endswith(".weight"):
            if data_torch.ndim == 3 and data_torch.shape[2] == 1:
                data_torch = data_torch.squeeze(2)

        if "depth_conv" in name and name.endswith(".weight"):
            if data_torch.ndim == 3 and data_torch.shape[1] == 1:
                data_torch = data_torch.squeeze(1)

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Granite4VisionForConditionalGeneration")
class Granite4VisionMmprojModel(MmprojModel):
    has_vision_encoder = True
    has_audio_encoder = False

    @staticmethod
    def get_normalized_projector_map(global_config: dict) -> list[tuple[int, int, str, int]]:
        """Normalize both deepstack and spatial projector maps to the form:
        (vision_layer, llm_layer, <type>, type_index)

        This is then used to populate the following mappings:
        - vision_feature_layers (mmproj hparam): ordered list of all
          vision_layer values where order corresponds with the order of the
          stacked projector tensors
          NOTE: Values may appear multiple times for spatial projectors
        - tensor_prefix_map (mmproj tensors): mapping from tensor prefixes to
          the index of the corresponding projector in the stacked tensors
        - deepstack_layer_arr (llm hparam): per-text-layer array indicating
          which input vision feature should be injected at that layer
          (-1 if none)

        Output: (vision_layer, llm_layer, <type>, type_index)
        """
        deepstack_map = global_config.get("deepstack_layer_map", [])  # [[vis_layer, llm_layer], ...]
        spatial_layers = global_config.get("spatial_target_layers", [])  # [llm_layer, ...]
        n_text_layers = global_config["text_config"]["num_hidden_layers"]
        n_vision_layers = global_config["vision_config"]["num_hidden_layers"]
        normalized_projector_map = []
        if deepstack_map:
            for deepstack_idx, (vision_layer, llm_layer) in enumerate(sorted(deepstack_map)):
                if vision_layer < 0:
                    vision_layer = n_vision_layers + vision_layer
                if llm_layer < 0:
                    llm_layer = n_text_layers + llm_layer
                normalized_projector_map.append((vision_layer, llm_layer, "layerwise", deepstack_idx))
        if spatial_layers:
            spatial_vision_layer = global_config.get("spatial_vision_layer", -1)
            if spatial_vision_layer < 0:
                spatial_vision_layer = n_vision_layers + spatial_vision_layer
            for spatial_idx, llm_layer in enumerate(spatial_layers):
                normalized_projector_map.append((spatial_vision_layer, llm_layer, "spatial", spatial_idx))
        return list(sorted(normalized_projector_map, key=(lambda entry: entry[1])))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        normalized_projector_map = self.get_normalized_projector_map(self.global_config)
        self._n_proj = len(normalized_projector_map)

        self._tensor_prefix_map = {
            f"model.{proj_type}_projectors.{type_idx}": proj_idx
            for proj_idx, (_, _, proj_type, type_idx) in enumerate(normalized_projector_map)
        }
        self._vision_feature_layers = [vision_layer for vision_layer, _, _, _ in normalized_projector_map]
        self._spatial_offsets = [
            type_idx if proj_type == "spatial" else -1
            for _, _, proj_type, type_idx in normalized_projector_map
        ]

    def set_gguf_parameters(self):
        assert self.hparams_vision is not None
        super().set_gguf_parameters()

        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.GRANITE4_VISION)

        # SigLIP encoder hparams
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams.get("layer_norm_eps", 1e-6))
        self.gguf_writer.add_vision_use_gelu(True)

        # Preprocessor
        self.gguf_writer.add_vision_preproc_image_size(self.hparams.get("image_size", 384))

        # QFormer projector config
        ds_rate = self.global_config["downsample_rate"]
        ds_parts = ds_rate.split("/")
        assert len(ds_parts) == 2, f"Invalid 'downsample_rate' value: {ds_rate}"
        query_side, window_side = [int(p) for p in ds_parts]
        self.gguf_writer.add_vision_projector_query_side(query_side)
        self.gguf_writer.add_vision_projector_window_side(window_side)

        # Set vision feature layers
        self.gguf_writer.add_vision_feature_layers(self._vision_feature_layers)

        # Set the spatial offests per projector
        self.gguf_writer.add_vision_spatial_offsets(self._spatial_offsets)

        # Add flattened image grind pinpoints (resolution candidates internally)
        if pinpoints := self.global_config.get("image_grid_pinpoints"):
            # Flatten with h, w -> w, h inversion
            pinpoints = [val for h, w in pinpoints for val in (w, h)]
            self.gguf_writer.add_vision_image_grid_pinpoints(pinpoints)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, _ = item
        if ("vision_model.head" in name or name.startswith("lm_head")):
            return None
        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:

        # Detect projector tensors and bin them
        projector_idx = None
        for prefix, proj_idx in self._tensor_prefix_map.items():
            if name.startswith(prefix):
                projector_idx = proj_idx
                break
        if projector_idx is not None:
            # If this projector tensor has a block id within the projector,
            # alias the bid to projector_idx
            #
            # TODO: currently, none of the Granite 4 Vision models have
            # projectors with multiple QFormer layers, so the `layer.{}` index
            # is always 0. This allows us to simply map to a single `bid` that
            # matches the projector index. If this changes, we'll need a
            # convention that merges the two IDs.
            id_matches = list(re.finditer(r"\.([0-9]+)\.", name))
            all_ids = [int(m.group(1)) for m in id_matches]
            assert len(all_ids) >= 1 and len(all_ids) <= 2, "Must have at least 1 and at most 2 ids in tensor names"
            # If not layer id, just use the projector index
            new_bid = projector_idx
            if len(all_ids) == 1:
                new_name = name[:id_matches[0].span(1)[0]] + str(new_bid) + name[id_matches[0].span(1)[1]:]
            else: # len(all_ids) == 2
                new_bid = projector_idx # + all_ids[1]
                new_name = name[:id_matches[0].span(0)[0]] + name[id_matches[0].span(1)[1]:id_matches[1].span(1)[0]] + str(new_bid) + name[id_matches[1].span(1)[1]:]
            yield from super().modify_tensors(data_torch, new_name, new_bid)
            return
        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("GraniteSwitchForCausalLM")
class GraniteSwitchModel(GraniteModel):
    """Conversion for IBM's GraniteSwitchForCausalLM with dynamic adapter selection."""
    model_arch = gguf.MODEL_ARCH.GRANITE_SWITCH

    def set_gguf_parameters(self):
        """Granite Switch extends Granite with adapter switching parameters."""
        super().set_gguf_parameters()

        # The switch layer is not a decoder layer - adjust block_count
        # HF num_hidden_layers includes the switch layer, but llama.cpp
        # expects block_count to be just the decoder layers
        if self.hparams.get("num_adapters", 0) > 0:
            self.hparams["num_hidden_layers"] = self.hparams.get("num_hidden_layers", 40) - 1
            self.block_count = self.hparams["num_hidden_layers"]
            logger.info("gguf: adjusted block_count to %d (subtracting switch layer)", self.block_count)

        config = self.hparams

        n_adapters = config.get("num_adapters", 0)
        if n_adapters == 0:
            raise ValueError("Not a Granite Switch model: num_adapters not found or is 0")

        self.gguf_writer.add_adapter_count(n_adapters)
        logger.info("gguf: (graniteswitch) adapter_count = %s", n_adapters)

        adapter_ranks = config.get("adapter_ranks", [])
        self.gguf_writer.add_adapter_ranks(adapter_ranks)
        logger.info("gguf: (graniteswitch) adapter_ranks = %s", adapter_ranks)

        max_lora_rank = max(adapter_ranks) if adapter_ranks else 0
        self.gguf_writer.add_max_lora_rank(max_lora_rank)
        logger.info("gguf: (graniteswitch) max_lora_rank = %s", max_lora_rank)

        adapter_token_ids = config.get("adapter_token_ids", [])
        self.gguf_writer.add_adapter_token_ids(adapter_token_ids)
        logger.info("gguf: (graniteswitch) adapter_token_ids = %s", adapter_token_ids)

        adapter_substitute_token_ids = config.get("adapter_substitute_token_ids", [])
        self.gguf_writer.add_adapter_substitute_token_ids(adapter_substitute_token_ids)
        logger.info("gguf: (graniteswitch) adapter_substitute_token_ids = %s", adapter_substitute_token_ids)

        control_token_gain = config.get("control_token_gain", 15.0)
        self.gguf_writer.add_control_token_gain(control_token_gain)
        logger.info("gguf: (graniteswitch) control_token_gain = %s", control_token_gain)

        switch_head_dim = config.get("switch_head_dim", 32)
        self.gguf_writer.add_switch_head_dim(switch_head_dim)
        logger.info("gguf: (graniteswitch) switch_head_dim = %s", switch_head_dim)

        projection_head_dim = config.get("projection_head_dim", 64)
        self.gguf_writer.add_projection_head_dim(projection_head_dim)
        logger.info("gguf: (graniteswitch) projection_head_dim = %s", projection_head_dim)


        lora_target_modules = config.get("lora_target_modules", [])
        self.gguf_writer.add_lora_target_modules(lora_target_modules)
        logger.info("gguf: (graniteswitch) lora_target_modules = %s", lora_target_modules)

    def prepare_tensors(self):
        """Prepare all tensors for writing, including fused B slice concatenation."""
        # First, collect all tensors
        all_tensors = []
        for new_name, data_torch in self.get_tensors():
   # Convert unsupported dtypes to float32 (same as parent implementation)
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)
            
            for result in self.modify_tensors(data_torch, new_name, None):
                if isinstance(result, tuple) and len(result) == 2:
                    nn, dt = result
                    # Ensure yielded tensors are also in supported dtype
                    if dt.dtype not in (torch.float16, torch.float32):
                        dt = dt.to(torch.float32)
                    all_tensors.append((nn, dt))
        
        # Add concatenated fused B tensors
        for (layer_id, module), adapters in self._fused_b_slices.items():
            module_map = {
                "qkv_proj": "attn_qkv",
                "input_linear": "ffn_gate_up",
            }
            gguf_module = module_map.get(module)
            if gguf_module:
                for gguf_name, data in self._yield_fused_b_tensors(layer_id, module, gguf_module):
                    if data.dtype not in (torch.float16, torch.float32):
                        data = data.to(torch.float32)
                    all_tensors.append((gguf_name, data))
        
        # Sort tensors by name for consistent output
        self_tensors = []
        for new_name, data_torch in sorted(all_tensors, key=lambda x: x[0]):
            self_tensors.append((new_name, data_torch))
        
        self._model_tensors = {self.model_arch: [t[0] for t in self_tensors]}
        
        for new_name, data_torch in self_tensors:
            self.gguf_writer.add_tensor(new_name, data_torch.numpy())

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        # Skip multimodal tensors
        if (
            name.startswith(("encoder."))
            or "image_" in name
            or "layerwise_projectors" in name
            or "spatial_projectors" in name
        ):
            return
        # Skip config tensors that are not model weights
        if name in ("model.adapter_token_ids",):
            return
        # Skip switch lookup table (can be derived from adapter_token_ids + adapter_substitute_token_ids)
        if name.endswith(".switch.control_to_substitute_lut"):
            return
        return super().filter_tensors(item)

  def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Handle Granite Switch tensor naming conventions."""
        # Handle fused module adapter tensors (qkv_proj, shared_mlp with slices)
        if re.search(r"\.lora_(?:A|B)_slices\.", name):
            yield from self._handle_fused_adapter_tensor(data_torch, name, bid)
            return
        # Check if this is a non-fused adapter tensor (e.g., o_proj.lora_A)
        if re.search(r"\.(?:o_proj|output_linear)\.(?:lora_A|lora_B)$", name):
            yield from self._handle_adapter_tensor(data_torch, name, bid)
            return
        # Handle base_layer weights for LoRA modules
        if ".base_layer.weight" in name:
            yield from self._handle_base_layer_tensor(data_torch, name, bid)
            return
        # Handle layer norms - map HF names to GGUF names with layer offset
        if "input_layernorm.weight" in name:
            layer_id = None
            for i, part in enumerate(name.split(".")):
                if part == "layers" and i + 1 < len(name.split(".")):
                    try:
                        layer_id = int(name.split(".")[i + 1])
                        break
                    except ValueError:
                        continue
            if layer_id is not None:
                # HF layer 0 -> GGUF layer 1 (switch layer takes GGUF layer 0)
                yield from [(f"blk.{layer_id + 1}.attn_norm.weight", data_torch)]
                return
        if "post_attention_layernorm.weight" in name:
            layer_id = None
            for i, part in enumerate(name.split(".")):
                if part == "layers" and i + 1 < len(name.split(".")):
                    try:
                        layer_id = int(name.split(".")[i + 1])
                        break
                    except ValueError:
                        continue
            if layer_id is not None:
                yield from [(f"blk.{layer_id + 1}.ffn_norm.weight", data_torch)]
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def _handle_base_layer_tensor(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Handle base_layer.weight tensors, mapping to standard tensor names."""
        parts = name.split(".")

        # Extract layer id
        layer_id = None
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_id = int(parts[i + 1])
                    break
                except ValueError:
                    continue

        if layer_id is None:
            logger.warning("Skipping base_layer tensor without valid layer_id: %s", name)
            return

        # Add offset of +1 because layer 0 in GGUF is reserved for the switch layer
        # HF layers 0-39 become GGUF layers 1-40
        layer_id += 1

        # Find the module name (the part before base_layer)
        module = None
        for i, part in enumerate(parts):
            if part == "base_layer":
                if i > 0:
                    module = parts[i - 1]
                break

        if module is None:
            logger.warning("Could not parse module from: %s", name)
            return

        # Map HuggingFace module names to GGUF tensor types
        module_to_tensor = {
            "qkv_proj": ("attn_qkv", None),
            "o_proj": ("attn_output", None),
            "input_linear": ("ffn_gate", "ffn_up"),
            "output_linear": ("ffn_down", None),
        }

        if module not in module_to_tensor:
            logger.warning("Unknown base_layer module %s, skipping", module)
            return

        gguf_prefix = module_to_tensor[module][0]
        gguf_suffix = module_to_tensor[module][1]

        if gguf_suffix:
            # For fused modules like input_linear (gate + up fused), we need to split
            # The tensor shape is [2 * intermediate_size, hidden_size]
            # Split into two tensors of [intermediate_size, hidden_size]
            mid_dim = data_torch.shape[0] // 2
            gate_data = data_torch[:mid_dim]
            up_data = data_torch[mid_dim:]
            yield f"blk.{layer_id}.ffn_gate.weight", gate_data
            yield f"blk.{layer_id}.ffn_up.weight", up_data
        else:
            yield f"blk.{layer_id}.{gguf_prefix}.weight", data_torch

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-collect fused adapter B slices for concatenation
        self._fused_b_slices = {}  # {(layer_id, module): {adapter_id: [slice_data, ...]}}

    def _collect_fused_b_slice(self, layer_id, module, adapter_id, slice_data):
        """Collect a B slice for later concatenation."""
        key = (layer_id, module)
        if key not in self._fused_b_slices:
            self._fused_b_slices[key] = {}
        if adapter_id not in self._fused_b_slices[key]:
            self._fused_b_slices[key][adapter_id] = []
        self._fused_b_slices[key][adapter_id].append(slice_data)

    def _yield_fused_b_tensors(self, layer_id, module, gguf_module):
        """Yield concatenated B tensors from collected slices."""
        key = (layer_id, module)
        if key not in self._fused_b_slices:
            return
        
        for adapter_id, slices in sorted(self._fused_b_slices[key].items()):
            # Concatenate slices along the output features dimension
            combined = torch.cat(slices, dim=0)  # [total_out_features, max_lora_rank]
            gguf_name = f"blk.{layer_id}.{gguf_module}.lora_b.adapter_{adapter_id}.weight"
            yield gguf_name, combined

    def _handle_fused_adapter_tensor(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Handle fused module adapter tensors (qkv_proj, shared_mlp with slices).
        
        For fused modules, Lora_A weights are shared across slices (only first slice has them).
        Lora_B weights have different output dimensions per slice and need to be concatenated.
        """
        parts = name.split(".")

       # Find the layer index
        layer_id = None
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_id = int(parts[i + 1])
                    break
                except ValueError:
                    continue

        if layer_id is None:
            logger.warning("Skipping fused adapter tensor without valid layer_id: %s", name)
            return

        # Layer 0 is the switch layer - skip adapter tensors for switch layer
        if layer_id == 0:
            logger.info("Skipping switch layer fused adapter tensor: %s", name)
            return

        # Add offset of +1 because layer 0 is reserved for the switch layer
        layer_id += 1

        # Determine if this is lora_A or lora_B, and the slice index
        lora_type = None
        slice_idx = None
        
        for i, part in enumerate(parts):
            if part in ("lora_A_slices", "lora_B_slices"):
                lora_type = "lora_a" if part == "lora_A_slices" else "lora_b"
                if i + 1 < len(parts):
                    try:
                        slice_idx = int(parts[i + 1])
                    except ValueError:
                        pass
                module = parts[i - 1] if i > 0 else None
                break

        if lora_type is None or slice_idx is None or module is None:
            logger.warning("Could not parse fused adapter tensor: %s", name)
            return

        # Map HuggingFace module names to GGUF module names
        module_map = {
            "qkv_proj": "attn_qkv",
            "input_linear": "ffn_gate_up",
        }

        gguf_module = module_map.get(module)
        if not gguf_module:
            logger.warning("Unknown fused module type %s, skipping", module)
            return

        if data_torch.ndim != 4:
            logger.warning("Unexpected tensor ndim %d for fused adapter tensor %s", data_torch.ndim, name)
            return

        num_adapters = data_torch.shape[0]
        
        if lora_type == "lora_a":
            # Only process the first slice (others are duplicates)
            if slice_idx != 0:
                return
            
            for adapter_idx in range(1, num_adapters):
                adapter_id = adapter_idx - 1
                adapter_data = data_torch[adapter_idx, 0]
                gguf_name = f"blk.{layer_id}.{gguf_module}.lora_a.adapter_{adapter_id}.weight"
                yield gguf_name, adapter_data
        else:
            # Collect B slices for later concatenation
            for adapter_idx in range(1, num_adapters):
                adapter_id = adapter_idx - 1
                slice_data = data_torch[adapter_idx, 0]  # [out_features_slice, max_lora_rank]
                self._collect_fused_b_slice(layer_id, module, adapter_id, slice_data)

    def _handle_adapter_tensor(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        """Convert and write adapter LoRA tensors.
        
        HF stores adapter weights in tensors with shape [num_adapters, 1, max_lora_rank, features]
        where each adapter index corresponds to a different LoRA adapter.
        We need to split these into individual tensors per adapter.
        """
        parts = name.split(".")

        # Find the layer index
        layer_id = None
        for i, part in enumerate(parts):
            if part == "layers" and i + 1 < len(parts):
                try:
                    layer_id = int(parts[i + 1])
                    break
                except ValueError:
                    continue

        if layer_id is None:
            logger.warning("Skipping adapter tensor without valid layer_id: %s", name)
            return

        # Layer 0 is the switch layer - skip adapter tensors for switch layer
        if layer_id == 0:
            logger.info("Skipping switch layer adapter tensor: %s", name)
            return

        # Add offset of +1 because layer 0 is reserved for the switch layer
        layer_id += 1

        # Determine lora type (a or b)
        lora_type = "lora_a" if "lora_A" in name else "lora_b"

        # Find the module name
        module = None
        for i, part in enumerate(parts):
            if part in ("lora_A_slices", "lora_B_slices", "lora_A", "lora_B"):
                if i > 0:
                    module = parts[i - 1]
                break

        if module is None:
            logger.warning("Could not parse module from: %s", name)
            return

        # Map HuggingFace module names to GGUF module names
        module_map = {
            "qkv_proj": "attn_qkv",
            "o_proj": "attn_output",
            "gate_proj": "ffn_gate",
            "up_proj": "ffn_up",
            "down_proj": "ffn_down",
            "gate_up_proj": "ffn_gate_up",
            "input_linear": "ffn_gate_up",
            "output_linear": "ffn_down",
        }

        gguf_module = module_map.get(module)
        if not gguf_module:
            logger.warning("Unknown module type %s, skipping adapter tensor", module)
            return

        # HF tensor shape: [num_adapters, 1, max_lora_rank, features] for lora_A
        # HF tensor shape: [num_adapters, 1, out_features, max_lora_rank] for lora_B
        # Index 0 is base model, indices 1+ are adapters (adapter_id = index - 1)
        if data_torch.ndim != 4:
            logger.warning("Unexpected tensor ndim %d for adapter tensor %s, skipping", data_torch.ndim, name)
            return

        num_adapters = data_torch.shape[0]
        
        # Write each adapter's weights as a separate tensor
        for adapter_idx in range(1, num_adapters):
            adapter_id = adapter_idx - 1  # HF index 0 = base, index 1 = adapter 0
            
            if lora_type == "lora_a":
                # Shape: [num_adapters, 1, max_lora_rank, features] -> [max_lora_rank, features]
                adapter_data = data_torch[adapter_idx, 0]
            else:
                # Shape: [num_adapters, 1, out_features, max_lora_rank] -> [out_features, max_lora_rank]
                adapter_data = data_torch[adapter_idx, 0]
            
            gguf_name = f"blk.{layer_id}.{gguf_module}.{lora_type}.adapter_{adapter_id}.weight"
            yield gguf_name, adapter_data
