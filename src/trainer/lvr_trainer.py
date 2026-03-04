import os
import torch
import torch.nn as nn
import wandb
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

#用于处理 DeepSpeed ZeRO-3 优化器下的参数收集：如果参数被 ZeRO-3 分区，先收集（gather）参数，将参数移到 CPU 并克隆，避免引用问题
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class QwenLVRSFTTrainer(Trainer):

    def __init__(self, *args, temp_folder=None, oci_handler=None, **kwargs):
        super(QwenLVRSFTTrainer, self).__init__(*args, **kwargs)
        # if online checkpointing
        if oci_handler:
            self.oci_handler = oci_handler
        # 修改，下面一行放到if外面
        self.temp_folder = temp_folder     # temp_file class; "/dockerx/Local/users/bangzheng/model_name/run_name-[random]"
        self._compression_steps = 0
        self._compression_before_sum = 0
        self._compression_after_sum = 0
        self._compression_ratio_sum = 0.0
        self._reduction_ratio_sum = 0.0

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        # 识别不同模块的参数，分别创建优化器
        #优化器分组策略：对每个模块分别创建带权重衰减和不带权重衰减的参数组。每个模块可以设置独立的 learning rate，支持 Adam8bit 优化器，对 Embedding 层使用 32-bit 精度
        if self.optimizer is None:
            #确定了哪些参数需要进行权重衰减。它排除了所有的 LayerNorm 层和所有包含 "bias" 字样的参数。
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []
            lvr_head_parameters =[]

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]
            if self.args.lvr_head_lr is not None:
                lr_mapper["lvr_head"] = self.args.lvr_head_lr
                lvr_head_parameters = [name for name, _ in opt_model.named_parameters() if "lvr_head" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters + lvr_head_parameters

                # 分组：视觉模块、merger模块、lvrhead模块、非特殊模块（分别是否衰减）
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
                
                if lvr_head_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in lvr_head_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.lvr_head_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in lvr_head_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.lvr_head_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    #8-bit优化器，对embedding层仍然使用32bit
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

#在本地保存检查点，上传到云端存储，清理本地临时文件以节省空间，管理检查点轮换（删除旧检查点）
    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        # modified to support online checkpointing
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        # output_dir is the local path forcheckpoint
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
            best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Save the Trainer state
        if self.args.should_save:
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # output_dir is local; now we save to cloud if needed
        if self.temp_folder:
            remote_chkpt_folder = os.path.join(self.args.remote_output_dir,checkpoint_folder)
            if remote_chkpt_folder[0] == '/':
                remote_chkpt_folder = remote_chkpt_folder[1:]       #remote pathing rules will take bucket//checkpoints, need to remove the dup
            self.oci_handler.save_checkpoint(output_dir,remote_chkpt_folder)    #save local chkpt to remote folder
            # remove the local 
            self.temp_folder.cleanup(checkpoint_name=checkpoint_folder)


        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # Solely rely on numerical checkpoint id for rotation.
            # mtime is not reliable especially on some fuse fs in cloud environments.
            self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

    def compute_loss(self, model, inputs,num_items_in_batch=None, return_outputs=False):

        if self.args.enable_data_packing:
            batch_size = inputs['input_ids'].size(0)
            total_tokens = inputs['input_ids'].size(0) * inputs['input_ids'].size(1)
            self.log({
            "batch_size": batch_size,
            "tokens_per_device": total_tokens,})

        outputs = model(**inputs)
        # loss = outputs.loss  # total loss
        loss_ce = outputs.loss_ce
        loss_lvr = outputs.loss_lvr
        loss_mode_switch = outputs.loss_mode_switch

        #celoss、lvrloss、switchloss
        if self.args.mode_switch_loss:
            loss = loss_ce + self.args.loss_lvr_lambda * loss_lvr + self.args.loss_mode_switch_lambda * loss_mode_switch
        else:
            loss = loss_ce + self.args.loss_lvr_lambda * loss_lvr if self.args.loss_lvr_lambda > 0 else loss_ce

        # Log each component
        self.log({
            "loss_total": loss.detach().item(),
            "loss_ce": loss_ce.detach().item(),
            "loss_lvr": loss_lvr.detach().item() if loss_lvr is not None else 0.0,
            "loss_mode_switch": loss_mode_switch.detach().item() if loss_mode_switch is not None else 0.0,
        })

        before_cnt = getattr(outputs, "lvr_tokens_before_count", None)
        after_cnt = getattr(outputs, "lvr_tokens_after_count", None)
        compression_ratio = getattr(outputs, "lvr_compression_ratio", None)
        reduction_ratio = getattr(outputs, "lvr_reduction_ratio", None)
        if (
            before_cnt is not None
            and after_cnt is not None
            and compression_ratio is not None
            and reduction_ratio is not None
        ):
            self._compression_steps += 1
            self._compression_before_sum += int(before_cnt)
            self._compression_after_sum += int(after_cnt)
            self._compression_ratio_sum += float(compression_ratio)
            self._reduction_ratio_sum += float(reduction_ratio)

            avg_compression_ratio = self._compression_ratio_sum / self._compression_steps
            avg_reduction_ratio = self._reduction_ratio_sum / self._compression_steps
            self.log({
                "lvr_tokens_before": int(before_cnt),
                "lvr_tokens_after": int(after_cnt),
                "lvr_compression_ratio": float(compression_ratio),
                "lvr_reduction_ratio": float(reduction_ratio),
                "lvr_compression_ratio_avg": float(avg_compression_ratio),
                "lvr_reduction_ratio_avg": float(avg_reduction_ratio),
            })


        return (loss, outputs) if return_outputs else loss

    def train(self, *args, **kwargs):
        train_output = super().train(*args, **kwargs)
        if self._compression_steps > 0:
            avg_compression_ratio = self._compression_ratio_sum / self._compression_steps
            avg_reduction_ratio = self._reduction_ratio_sum / self._compression_steps
            total_before = self._compression_before_sum
            total_after = self._compression_after_sum
            global_compression_ratio = (total_after / total_before) if total_before > 0 else 1.0
            global_reduction_ratio = 1.0 - global_compression_ratio

            summary = (
                f"[LVR Compression Summary] steps={self._compression_steps}, "
                f"before_sum={total_before}, after_sum={total_after}, "
                f"avg_ratio={avg_compression_ratio:.6f}, avg_reduction={avg_reduction_ratio:.6f}, "
                f"global_ratio={global_compression_ratio:.6f}, global_reduction={global_reduction_ratio:.6f}"
            )
            logger.info(summary)
            if self.is_world_process_zero():
                print(summary)
                self.log({
                    "lvr_compression_ratio_avg_final": float(avg_compression_ratio),
                    "lvr_reduction_ratio_avg_final": float(avg_reduction_ratio),
                    "lvr_compression_ratio_global_final": float(global_compression_ratio),
                    "lvr_reduction_ratio_global_final": float(global_reduction_ratio),
                    "lvr_tokens_before_sum_final": int(total_before),
                    "lvr_tokens_after_sum_final": int(total_after),
                })

        return train_output
