# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import copy
import enum
import logging
import re
from functools import lru_cache
from typing import Optional

from megatron.core import parallel_state


class LayerType(enum.Enum):
    """Layer type used by custom pipeline parallel layout."""

    embedding = 1
    loss = 2
    encoder = 3
    decoder = 4
    mtp = 5


logger = logging.getLogger(__name__)


class PipelineParallelLayerLayout:
    """Configuration of custom pipeline parallel layer partitioning."""

    def __repr__(self) -> str:
        if isinstance(self.input_data, str):
            return self.input_data
        return str(self.input_data)

    def __init__(self, layout: str | list, pipeline_model_parallel_size: int):
        """Initialize PipelineParallelLayerLayout from a list or a str.

        Format validation will be done here.
        """

        self.input_data = layout
        if isinstance(layout, str):
            layout = PipelineParallelLayerLayout.parse_str_to_list(layout)
        else:
            layout = copy.deepcopy(layout)
        assert all(isinstance(row, list) for row in layout), (
            f"pipeline_model_parallel_layout must be a list of lists, but got {[type(row) for row in layout]=}"
        )

        # Check PP size and get VPP size.
        assert len(layout) % pipeline_model_parallel_size == 0, (
            f"pipeline_model_parallel_layout must be divisible"
            f" by pipeline_model_parallel_size ({len(layout)=},"
            f" {pipeline_model_parallel_size=})"
        )
        virtual_pipeline_model_parallel_size = len(layout) // pipeline_model_parallel_size

        # Convert 1D layout to 2D layout.
        layout = [
            [
                layout[vpp_rank * pipeline_model_parallel_size + pp_rank]
                for vpp_rank in range(virtual_pipeline_model_parallel_size)
            ]
            for pp_rank in range(pipeline_model_parallel_size)
        ]

        # Convert all strings in pipeline_model_parallel_layout to LayerType.
        for pp_rank in range(pipeline_model_parallel_size):
            for vpp_rank in range(virtual_pipeline_model_parallel_size):
                transferred_layout = []
                for layer_type in layout[pp_rank][vpp_rank]:
                    assert isinstance(layer_type, (LayerType, str)), (
                        f"elements in pipeline_model_parallel_layout must be LayerType or str,"
                        f" but got {type(layer_type)}."
                    )
                    if isinstance(layer_type, str):
                        layer_type = layer_type.strip().lower()
                        assert layer_type in LayerType.__members__, f"{layer_type} is not a valid LayerType"
                        layer_type = LayerType[layer_type]
                    transferred_layout.append(layer_type)
                layout[pp_rank][vpp_rank] = transferred_layout

        # Flatten the pipeline layout in layer id order.
        flatten_layout = []
        for vpp_rank in range(virtual_pipeline_model_parallel_size):
            for row in layout:
                flatten_layout.extend(row[vpp_rank])

        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        self.layout = layout
        self.flatten_layout = flatten_layout

    def validate_layer_layout(self, num_layers: int, mtp_num_layers: int):
        """Check whether the layout is valid."""

        # Check whether the input layer id is valid.
        assert all(isinstance(x, LayerType) for x in self.flatten_layout), "All layers must be a valid LayerType."

        # Embedding layer and loss layer must be specified.
        assert self.flatten_layout[0] == LayerType.embedding, (
            f"The first layer must be embedding, but got {self.flatten_layout[0]}"
        )
        assert self.flatten_layout[-1] == LayerType.loss, (
            f"The last layer must be loss, but got {self.flatten_layout[-1]}"
        )

        # Layer number verification.
        assert self.flatten_layout.count(LayerType.embedding) == 1, "Embedding must be specified exactly once"
        assert self.flatten_layout.count(LayerType.loss) == 1, "Loss must be specified exactly once"
        assert self.flatten_layout.count(LayerType.decoder) == num_layers, (
            f"Number of decoder layers {self.flatten_layout.count(LayerType.decoder)}must match num_layers {num_layers}"
        )

        # MTP layer verification.
        assert self.flatten_layout.count(LayerType.mtp) == mtp_num_layers or (
            mtp_num_layers is None and self.flatten_layout.count(LayerType.mtp) == 0
        ), "Number of mtp layers in layout must match mtp_num_layers"
        for i in range(len(self.flatten_layout)):
            if self.flatten_layout[i] == LayerType.mtp:
                assert self.flatten_layout[i:].count(LayerType.decoder) == 0, (
                    "decoder layers must be placed before MTP layers"
                )
                break
        for pp_rank in range(self.pipeline_model_parallel_size):
            for vpp_rank in range(self.virtual_pipeline_model_parallel_size - 1):
                assert LayerType.mtp not in self.layout[pp_rank][vpp_rank], (
                    f"Currently we restrict that the MTP should be always in the last "
                    f"virtual pipeline stage of that rank. But got {self.layout[pp_rank][vpp_rank]}"
                )
        for pp_rank in range(self.pipeline_model_parallel_size):
            if LayerType.mtp in self.layout[pp_rank][-1]:
                assert self.layout[pp_rank][-1].count(LayerType.mtp) == mtp_num_layers, (
                    "All of the MTP layers must be in the same one virtual pipeline stage"
                )
        for vpp_rank in range(self.virtual_pipeline_model_parallel_size - 1):
            assert LayerType.mtp not in self.layout[0][vpp_rank], (
                f"Currently we restrict that the MTP should not be in the first pp rank."
                f"But got {self.layout[0]} for the first pp rank."
            )

        # Detect MTP standalone usage.
        mtp_standalone = False
        for pp_rank in range(self.pipeline_model_parallel_size):
            if LayerType.mtp in self.layout[pp_rank][-1] and pp_rank != self.pipeline_model_parallel_size - 1:
                mtp_standalone = True
                break

        if self.flatten_layout.count(LayerType.encoder) > 0:
            raise NotImplementedError("Encoder layer is not supported for flexible pipeline layout")

        return mtp_standalone

    def _get_vp_stage(self, vp_stage: Optional[int]):
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            if vp_stage is None:
                vp_stage = parallel_state.get_virtual_pipeline_model_parallel_rank()
            assert vp_stage is not None, "vp_stage must be passed if virtual pipeline is enabled"
        else:
            vp_stage = 0
        return vp_stage

    def get_num_layers_to_build(
        self,
        layer_type: LayerType = LayerType.decoder,
        vp_stage: Optional[int] = None,
        pp_rank: Optional[int] = None,
    ):
        """Get the number of layers to build in the pipeline stage."""
        if pp_rank is None:
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        vp_stage = self._get_vp_stage(vp_stage)

        # Count layer numbers in this stage.
        num_layers_to_build = self.layout[pp_rank][vp_stage].count(layer_type)
        return num_layers_to_build

    def get_layer_offset(
        self,
        layer_type: LayerType = LayerType.decoder,
        vp_stage: Optional[int] = None,
        pp_rank: Optional[int] = None,
    ):
        """Get the layer offset in the pipeline stage."""
        if pp_rank is None:
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        vp_stage = self._get_vp_stage(vp_stage)

        # Calculate the offset by summing up the number of
        # layers in all the previous pipeline stages.
        offset = 0
        for _vpp_rank in range(vp_stage + 1):
            for _pp_rank in range(self.pipeline_model_parallel_size if _vpp_rank < vp_stage else pp_rank):
                offset += self.layout[_pp_rank][_vpp_rank].count(layer_type)
        return offset

    def get_layer_id_list(
        self,
        layer_type: LayerType = LayerType.decoder,
        vp_stage: Optional[int] = None,
        pp_rank: Optional[int] = None,
    ):
        """Get the list of layer_id for each layer in the pipeline stage."""
        offset = self.get_layer_offset(layer_type=layer_type, vp_stage=vp_stage, pp_rank=pp_rank)
        num_layers_to_build = self.get_num_layers_to_build(layer_type=layer_type, vp_stage=vp_stage, pp_rank=pp_rank)
        return list(range(offset, offset + num_layers_to_build))

    @staticmethod
    def _format_stage(stage):
        stage_repr = []
        prev_layer, prev_layer_cnt = None, 0
        for layer_type in stage + [None]:
            if layer_type == prev_layer:
                prev_layer_cnt += 1
            else:
                if prev_layer_cnt > 1:
                    stage_repr.append(f"{prev_layer.name}*{prev_layer_cnt}")
                elif prev_layer_cnt == 1:
                    stage_repr.append(f"{prev_layer.name}")
                prev_layer, prev_layer_cnt = layer_type, 1
        if len(stage_repr) == 0:
            stage_repr.append("(empty stage)")
        return ",".join(stage_repr)

    def pretty_repr(self):
        """Pretty representation of the custom layout, showing layers held by each stage."""

        matrix = []
        if self.virtual_pipeline_model_parallel_size > 1:
            header = [""] + [f"VPP rank {vpp_rank}" for vpp_rank in range(self.virtual_pipeline_model_parallel_size)]
            matrix.append(header)

        prev_row_repr, prev_row_start_pp_rank = None, None
        for pp_rank in range(self.pipeline_model_parallel_size + 1):
            row_repr = []
            if pp_rank < self.pipeline_model_parallel_size:
                for vpp_rank in range(self.virtual_pipeline_model_parallel_size):
                    stage = self.layout[pp_rank][vpp_rank]
                    row_repr.append(PipelineParallelLayerLayout._format_stage(stage))

            if row_repr != prev_row_repr:
                if prev_row_start_pp_rank == pp_rank - 1:
                    matrix.append([f"PP rank {pp_rank - 1}"] + prev_row_repr)
                elif prev_row_repr is not None:
                    matrix.append([f"PP rank {prev_row_start_pp_rank}-{pp_rank - 1}"] + prev_row_repr)
                prev_row_repr, prev_row_start_pp_rank = row_repr, pp_rank

        # Indent the matrix to make it more readable.
        lens = [max(map(len, col)) for col in zip(*matrix)]
        indents = 8 if self.virtual_pipeline_model_parallel_size <= 4 else 4
        fmt = (" " * indents).join("{{:{}}}".format(x) for x in lens)
        return "\n".join([fmt.format(*row) for row in matrix])

    @staticmethod
    @lru_cache()
    def from_str(layout, pipeline_model_parallel_size):
        """Parse the pipeline model parallel layout from a string."""
        parsed_layout = PipelineParallelLayerLayout(layout, pipeline_model_parallel_size)
        # Pretty print the layout distribution.
        from megatron.core.utils import log_single_rank

        log_single_rank(
            logger,
            logging.INFO,
            f"Parse pipeline model parallel layout {layout} to:\n" + parsed_layout.pretty_repr(),
        )
        return parsed_layout

    @staticmethod
    def get_num_stages_from_str(layout: str):
        """Get the number of PP * VPP stages from a layout string."""
        layout_list = PipelineParallelLayerLayout.parse_str_to_list(layout)
        return len(layout_list)

    @staticmethod
    def parse_str_to_list(layout_str: str):
        """Parse a layout string to a list of lists.

        Example: "Ettt|(tt|)*29,m|L" will be parsed to
        [["E","t","t","t"]]+[["t","t"]]*29+[["m"],["L"]].
        """

        layout_str = layout_str.replace(",", "")  # remove purely cosmetic commas

        # Unroll multiplications in the expression.
        patterns = [
            # Unroll expression in parentheses ()*n. Examples:
            # xy(ab|cd|ef)*2,pq -> xyab|cd|efab|cd|efpq
            # (ab)*3 -> ababab
            # ab,(cd|)*2 -> abcd|cd|
            # (|ab)*2,cd -> |ab|abcd
            r"\(([^)]+)\)\*(\d+)",
            r"(.)\*(\d+)",  # unroll x*n to n xs
        ]
        for pattern in patterns:
            layout_str = re.sub(pattern, lambda x: x.group(1) * int(x.group(2)), layout_str)

        char2layer_type = {
            "E": LayerType.embedding,
            "L": LayerType.loss,
            "t": LayerType.decoder,  # t denotes "transformer"
            "m": LayerType.mtp,
        }

        # Parse the layout string.
        layout_list = []
        for stage in layout_str.split("|"):
            layout_list.append([])
            for layer_char in stage:
                assert layer_char in char2layer_type, (
                    f"Invalid layer character: {layer_char} ({stage=}, {layout_str=}),"
                    f" known layer characters: {list(char2layer_type.keys())}"
                )

                layout_list[-1].append(char2layer_type[layer_char])
        return layout_list
