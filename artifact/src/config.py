import logging
import os
from typing import Any, Dict, List, Mapping, Optional, TypeAlias

from dataclasses import asdict, dataclass, field, is_dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf


DEFAULT_HOST_PATH = (
    "$HOME/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
)


def plain_config_dict(
    value: Any,
    *,
    error_label: str | None = None,
    resolve: bool = True,
) -> dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=resolve)
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, dict):
        if error_label:
            raise TypeError(f"{error_label} must resolve to a mapping")
        return {}
    return {str(key): item for key, item in value.items()}


def resolve_default_host_path(home: str | None = None) -> str:
    base_home = home or os.getenv("HOME", "")
    if not base_home:
        return DEFAULT_HOST_PATH
    return DEFAULT_HOST_PATH.replace("$HOME", base_home)


@dataclass
class SSHConfig:
    user: str = MISSING
    key_path: str = MISSING
    # Comma-separated host:port endpoints for round-robin remote SSH sessions.
    servers: str = MISSING
    # PATH for host commands (SSH doesn't source shell profile)
    host_path: str = DEFAULT_HOST_PATH


@dataclass(frozen=True)
class SSHEndpoint:
    host: str
    port: int


def resolve_ssh_endpoints(ssh_cfg: SSHConfig) -> list[SSHEndpoint]:
    raw_servers = str(ssh_cfg.servers).strip()
    if not raw_servers or raw_servers == str(MISSING):
        raise ValueError("ssh.servers must include at least one host:port endpoint")

    endpoints: list[SSHEndpoint] = []
    for raw_server in raw_servers.split(","):
        server = raw_server.strip()
        if not server:
            continue
        host, separator, port = server.rpartition(":")
        if not separator or not host or not port:
            raise ValueError(
                f"Invalid SSH server endpoint {server!r}; expected host:port"
            )
        endpoints.append(SSHEndpoint(host=host, port=int(port)))

    if not endpoints:
        raise ValueError("ssh.servers must include at least one host:port endpoint")
    return endpoints


@dataclass
class ScenarioConfig:
    name: str = "01_vuln_suid_gtfo"
    backend: str = "remote_ssh"
    base_command_timeout: int = 1
    slow_command_timeout: int = 10
    max_command_timeout: int = 90
    max_parallel_tool_calls: int = 8
    term_cols: int = 80
    term_rows: int = 24
    container_user: str = "lowpriv"
    container_password: str = "trustno1"
    extra_args: List[str] = field(default_factory=list)
    pre_command: Optional[str] = None
    command: Optional[str] = None
    ready_command: Optional[str] = None
    ready_command_timeout: Optional[int] = None
    log_message: Optional[str] = None
    use_registry_mirror: bool = False
    manual_sshd_start: bool = False
    auth_connect_timeout: int = 15
    image: Optional[str] = None
    setup_script: Optional[str] = None
    solution: Optional[Dict[str, Any]] = None


@dataclass
class AgentContextConfig:
    enabled: bool = True
    max_len: int = 32768
    reserve_ratio: float | None = None


@dataclass
class AgentRuntimeConfig:
    model_invoke_timeout: int = 180
    debug_timing_logs: bool = False


@dataclass
class AgentConfig:
    api_key: str = MISSING
    api_base: str = MISSING
    model: str = MISSING
    max_turns: int = MISSING
    params: Optional[Dict[str, Any]] = None
    context_management: AgentContextConfig = field(default_factory=AgentContextConfig)
    runtime: AgentRuntimeConfig = field(default_factory=AgentRuntimeConfig)


@dataclass
class SourceConfig:
    """Configuration for scenario source (static or procedural)."""

    type: str = "static"
    scenarios: Optional[List[str]] = None  # For static: list of scenarios
    seed: int = 42  # For procedural: base seed
    random_seed: bool = False  # For procedural trace collection: use random seeds
    randomize_env: bool = True  # For procedural: vary env content per run; False = same env per generator (ablation)
    generator_profile: Optional[str] = None  # Optional procedural generator profile override
    generators: List[str] = field(
        default_factory=list
    )  # For procedural: generator names


@dataclass
class RunnerConfig:
    mode: str = MISSING
    runs_per_item: int = MISSING
    max_runs: int | None = None
    output_dir: str = MISSING
    workers: int = 1  # Number of parallel workers (1 = sequential)
    max_retries_per_run: int = 1
    max_infra_retries_per_run: int = 3
    run_wall_clock_timeout: int | None = (
        3600  # Hard per-attempt wall-clock limit (seconds)
    )
    environment_setup_timeout: int | None = 120
    source: SourceConfig = field(default_factory=SourceConfig)


@dataclass
class PromptsConfig:
    system_template: str = MISSING
    start_instruction: str = MISSING
    no_tool_calls_nudge: str = MISSING
    template_vars: Dict[str, Any] = MISSING


@dataclass
class RLDatasetConfig:
    output_dir: str = MISSING
    write_test: bool = MISSING
    train_multiplier: int = MISSING
    test_multiplier: int = MISSING


@dataclass
class SFTSubsampleConfig:
    n: Optional[int] = None
    source_profile: str = "standard"
    output_profile: Optional[str] = None
    regime: str = "unguided"
    teacher: str = "deepseek"
    reasoning_variant: str = "long_reasoning"


@dataclass
class SFTDatasetConfig:
    profile: str = "standard"
    regime: str = "unguided"
    teacher: str = "deepseek"
    reasoning_variant: str = "long_reasoning"
    teacher_model: Optional[str] = None
    model_name: Optional[str] = None
    output_dir: str = MISSING
    validation_output_dir: Optional[str] = None
    trace_root: Optional[str] = None
    source_dir: Optional[str] = (
        None  # Explicit traces dir (e.g., "training" or "validation")
    )
    exclude_source_dir: Optional[str] = (
        None  # Optional split to exclude by (generator_name, seed)
    )
    max_per_generator: Optional[int] = (
        None  # Cap passing traces per generator (prune extras if enabled)
    )
    fail_on_split_collision: bool = False
    no_plots: bool = MISSING
    prune: bool = False
    subsample: SFTSubsampleConfig = field(default_factory=SFTSubsampleConfig)
    quality: "SFTQualityFilterConfig" = field(
        default_factory=lambda: SFTQualityFilterConfig()
    )


@dataclass
class SFTQualityFilterConfig:
    # Turn / size limits
    max_turns: int = 15
    min_turns: int = 2
    max_tokens: int = 32768

    # Reasoning quality
    min_reasoning_length: int = 60
    max_nudges: int = 0

    # Leakage and prompt integrity
    warn_on_secret_solution_leakage: bool = True
    reject_on_secret_solution_leakage: bool = True
    reject_on_holdout_leakage: bool = True
    verify_solution_in_prompt: bool = True

    # Individual filter toggles (transparency)
    reject_on_failure: bool = True
    reject_on_empty_reasoning: bool = True
    reject_on_invalid_tool_call_schema: bool = True
    reject_on_html_entities: bool = True
    reject_on_container_misconfig: bool = True
    reject_tool_call_free_turns: bool = False
    live_turn_max_attempts: int = 1

    # Container sanity
    container_expected_binaries: List[str] = field(
        default_factory=lambda: [
            "file",
            "strings",
            "ss",
            "ip",
            "ifconfig",
            "netstat",
            "getcap",
            "getpcaps",
            "hexdump",
            "lsb_release",
        ]
    )

    # Hidden solution leakage keywords
    secret_solution_leakage_keywords: List[str] = field(
        default_factory=lambda: [
            "internal solution",
            "based on the solution",
            "the solution data",
            "internal guidance",
            "vulnerability guidance",
            "the guidance",
            "hint suggests",
            "hint strongly suggests",
            "suggested by the hint",
            "hint says",
            "hint mentions",
            "hint mentioned",
            "hint indicates",
            "hint about",
            "hint/info provided",
            "hint in the solution",
            "hint in the challenge guidance",
            "description says",
            "challenge description mentions",
            "as hinted",
            "the solution suggests",
            "according to the hint",
            "secret hint",
            "secret hints",
            "secret data hint",
            "secret data hints",
            "solution hint",
            "solution hints",
            "scenario hint",
            "scenario hints",
            "scenario mentions",
            "scenario is about",
            "hint in the secret data",
            "the hint in the secret data",
            "hint from the secret data",
            "hint in the internal data",
            "hint from the internal data",
            "the hint from the internal data",
            "secret data says",
            "secret data mentions",
            "secret data suggests",
            "secret guidance",
            "secret solution",
            "secret solution data",
            "solution data",
            "scenario description",
            "solution:",
            "answer:",
            "step-by-step solution",
            "reference solution",
            "solution content",
            "solution steps",
        ]
    )

    # Reject HTML entity corruption in commands / reasoning
    reject_html_entities: List[str] = field(
        default_factory=lambda: ["&amp;", "&lt;", "&gt;", "&#38;", "&#60;", "&#62;"]
    )


@dataclass
class DatasetAuditConfig:
    experiment_id: str = "00_split_and_leakage_audit"
    condition: str = "split_leakage"
    stage: str = "audit"
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    run_id: str = "latest"
    output_dir: Optional[str] = None
    trace_root: str = "outputs/traces/trace_collection/standard"
    trace_leakage_glob: str = "**/stats/trace_leakage.jsonl"
    dataset_root: str = "outputs/data/privesc_sft/standard"
    dataset_glob: str = "**/traces.jsonl"
    paper_runs_root: str = "outputs/runs/paper"
    training_generator_profile: str = "training"
    validation_generator_profile: str = "holdout"
    procedural_validation_experiment: str = "eval/paper_procedural"
    static_benchmark_experiment: str = "eval/benchmark"
    static_paper_experiment_prefix: str = "eval/paper_static"
    model_visible_secret_keywords: List[str] = field(
        default_factory=lambda: [
            "internal solution",
            "the solution data",
            "internal guidance",
            "vulnerability guidance",
            "secret data says",
            "secret data suggests",
            "secret guidance",
            "secret solution",
            "secret solution data",
            "solution data",
            "step-by-step solution",
            "reference solution",
            "solution content",
            "solution steps",
        ]
    )


@dataclass
class DatasetsConfig:
    rl: RLDatasetConfig = field(default_factory=RLDatasetConfig)
    sft: SFTDatasetConfig = field(default_factory=SFTDatasetConfig)
    audit: DatasetAuditConfig = field(default_factory=DatasetAuditConfig)


@dataclass
class SFTReasoningVariantDerivationConfig:
    variant: str = "no_reasoning"
    source_variant: str = "long_reasoning"
    source_trace_root: Optional[str] = None
    output_trace_root: Optional[str] = None
    workers: int = 1
    incremental: bool = False


@dataclass
class SFTConfig:
    """Common SFT hyperparameters shared across trainers."""

    leave_out_scenario: Optional[str] = None

    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_seq_length: int = 32768
    lora_rank: int = 64
    lora_alpha: int = 32
    output_dir: str = "outputs/models/sft"
    run_dir: Optional[str] = None

    # Global effective batch size (examples per optimizer step). Unsloth DDP
    # derives per-process gradient accumulation from this value and world_size.
    batch_size: int = 8

    num_train_epochs: float = 10.0
    learning_rate: float = 4.9e-4

    seed: int = 1337

    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)

    trl: "SFTTRLConfig" = field(default_factory=lambda: SFTTRLConfig())
    unsloth: "SFTUnslothConfig" = field(default_factory=lambda: SFTUnslothConfig())


@dataclass
class SFTUnslothConfig:
    """Unsloth / HuggingFace trainer-specific SFT configuration."""

    wandb_project: str = "privesc-llm-unsloth-sft"
    wandb_run_name: str | None = None

    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    use_lora: bool = True

    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    )
    use_gradient_checkpointing: Optional[str] = "unsloth"  # or None/False
    adapter_only_lm_head: bool = False
    ensure_weight_tying: bool = False

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    warmup_steps: int = 0
    eval_strategy: str = "epoch"
    eval_steps: int | None = None
    eval_on_start: bool = False
    prediction_loss_only: bool = False
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    lr_scheduler_type: str = "linear"
    report_to: Any = "wandb"  # HF-style backend selector
    save_strategy: str = "epoch"
    save_steps: int = 1

    # Optional override to bound training for smoke tests.
    # When set, this is passed through to TRL/HF TrainingArguments.
    max_steps: int | None = None

    packing: bool = False
    dataset_num_proc: int = 4


@dataclass
class SFTTRLConfig:
    """TRL + PEFT trainer-specific SFT configuration.

    Uses standard HuggingFace transformers model loading with PEFT LoRA.
    """

    wandb_project: str = "privesc-llm-trl-sft"

    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    )
    gradient_checkpointing: bool = True

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    dataset_num_proc: int = 4

    max_train_examples: Optional[int] = None
    max_eval_examples: Optional[int] = None

    warmup_steps: int = 0
    eval_strategy: str = "epoch"
    eval_steps: int | None = None
    logging_steps: int = 1
    optim: str = "adamw_torch"
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    lr_scheduler_type: str = "linear"
    report_to: Any = "wandb"
    save_strategy: str = "epoch"
    save_steps: int = 1
    bf16: bool = True


@dataclass
class PasswordReusePatternConfig:
    name: str = ""
    add_service_distractor: bool = False


@dataclass
class PasswordReuseGeneratorConfig:
    patterns: List[PasswordReusePatternConfig] = field(default_factory=list)
    service_users: List[str] = field(default_factory=list)


@dataclass
class WeakPasswordPatternConfig:
    name: str = ""
    target_role: str = "root"
    password_source: str = "common_password"


@dataclass
class WeakPasswordGeneratorConfig:
    patterns: List[WeakPasswordPatternConfig] = field(default_factory=list)
    common_passwords: List[str] = field(default_factory=list)
    uid0_users: List[str] = field(default_factory=list)
    decoy_users: List[str] = field(default_factory=list)
    decoy_count_min: int = 1
    decoy_count_max: int = 2


@dataclass
class SudoGtfobinsGeneratorConfig:
    grant_subjects: List[str] = field(default_factory=list)


@dataclass
class SuidLayoutConfig:
    decoy_path_templates: List[str] = field(default_factory=list)
    service_names: List[str] = field(default_factory=list)
    decoy_source_paths: List[str] = field(default_factory=list)
    decoy_count_min: int = 0
    decoy_count_max: int = 0
    decoy_binaries: List[str] = field(default_factory=list)


@dataclass
class SudoLayoutConfig:
    sudoers_file_names: List[str] = field(default_factory=list)
    decoy_path_templates: List[str] = field(default_factory=list)
    decoy_wrappers: List[str] = field(default_factory=list)
    decoy_count_min: int = 0
    decoy_count_max: int = 0


@dataclass
class CronWildcardConfig:
    archive_path: str = ""
    job_directory: str = ""
    job_names: List[str] = field(default_factory=list)
    job_path: str = ""
    wait_seconds: int = 0
    exploit_script_name: str = ""


@dataclass
class CronWritableScriptConfig:
    job_directory: str = ""
    job_names: List[str] = field(default_factory=list)
    job_path: str = ""
    wait_seconds: int = 0
    script_mode: str = ""
    initial_script_lines: List[str] = field(default_factory=list)


@dataclass
class CredentialArtifactHistoryNoiseConfig:
    prefix_count_min: int = 0
    prefix_count_max: int = 0
    suffix_start: int = 0
    suffix_count_min: int = 0
    suffix_count_max: int = 0


@dataclass
class CredentialArtifactPlacementLocationsConfig:
    mail: List[str] = field(default_factory=list)
    config: List[str] = field(default_factory=list)
    note: List[str] = field(default_factory=list)


@dataclass
class PasswordFileGeneratorConfig:
    hint: str = ""
    file_templates: List[dict[str, str]] = field(default_factory=list)
    placement_locations: CredentialArtifactPlacementLocationsConfig = field(
        default_factory=CredentialArtifactPlacementLocationsConfig
    )


@dataclass
class PasswordHistoryGeneratorConfig:
    hint: str = ""
    history_filenames: List[str] = field(default_factory=list)
    history_templates: List[str] = field(default_factory=list)
    history_noise_commands: List[str] = field(default_factory=list)
    history_noise: CredentialArtifactHistoryNoiseConfig = field(
        default_factory=CredentialArtifactHistoryNoiseConfig
    )


@dataclass
class SshKeyReuseGeneratorConfig:
    ssh_target: str = ""


@dataclass
class GeneratorsConfig:
    """Configuration for procedural scenario generators.

    Values are loaded from conf/generators/default.yaml by Hydra.
    These define the pools of options each generator samples from.
    Holdout items (used in static benchmark) are excluded to prevent data leakage.
    """

    decoys_enabled: bool = False

    # Credential artifact generators
    password_file: PasswordFileGeneratorConfig = field(
        default_factory=PasswordFileGeneratorConfig
    )
    password_history: PasswordHistoryGeneratorConfig = field(
        default_factory=PasswordHistoryGeneratorConfig
    )

    # Password Reuse Generator
    password_reuse: PasswordReuseGeneratorConfig = field(
        default_factory=PasswordReuseGeneratorConfig
    )

    # Weak Password Generator
    weak_password: WeakPasswordGeneratorConfig = field(
        default_factory=WeakPasswordGeneratorConfig
    )

    # SSH Key Reuse Generator
    ssh_key_configs: List[dict[str, str]] = field(default_factory=list)
    user_ssh_dir_suffixes: List[str] = field(default_factory=list)
    shared_ssh_dirs: List[str] = field(default_factory=list)
    ssh_key_reuse: SshKeyReuseGeneratorConfig = field(
        default_factory=SshKeyReuseGeneratorConfig
    )

    # Cron Writable Script Generator
    cron_script_names: List[str] = field(default_factory=list)
    cron_writable_script: CronWritableScriptConfig = field(
        default_factory=CronWritableScriptConfig
    )

    # Cron Wildcard Generator
    cron_backup_dirs: List[str] = field(default_factory=list)
    cron_wildcard: CronWildcardConfig = field(default_factory=CronWildcardConfig)

    # Procedural low-priv username generation
    lowpriv_first_names: List[str] = field(default_factory=list)
    lowpriv_last_names: List[str] = field(default_factory=list)
    lowpriv_username_templates: List[str] = field(default_factory=list)

    # GTFOBins allowlists (loaded from conf/generators/*.yaml)
    capabilities_allowlist: List[str] = field(default_factory=list)
    suid_allowlist: List[str] = field(default_factory=list)
    suid_layout: SuidLayoutConfig = field(default_factory=SuidLayoutConfig)
    sudo_gtfobins: SudoGtfobinsGeneratorConfig = field(
        default_factory=SudoGtfobinsGeneratorConfig
    )
    sudo_allowlist: List[str] = field(default_factory=list)
    sudo_layout: SudoLayoutConfig = field(default_factory=SudoLayoutConfig)


@dataclass
class GTFOBinsConfig:
    """Configuration for GTFOBins extraction pipeline.

    Values are loaded from conf/gtfobins/default.yaml by Hydra.
    The defaults here are fallbacks for standalone usage.
    """

    gtfobins_dir: str = "external/GTFOBins/_gtfobins"
    output_dir: str = "conf/gtfobins/catalog"
    contexts: List[str] = field(
        default_factory=lambda: ["suid", "sudo", "capabilities"]
    )
    suid_allowlist: List[str] = field(default_factory=list)
    sudo_allowlist: List[str] = field(default_factory=list)
    extra_binaries: List[str] = field(default_factory=list)


_DEFAULT_RL_GENERATORS: list[str] = [
    "capabilities_gtfobins",
    "suid_gtfobins",
    "sudo_gtfobins",
    "cron_wildcard",
    "cron_writable_script",
    "password_file",
    "password_history",
    "password_reuse",
    "ssh_key_reuse",
    "weak_password",
]


def _default_procedural_source_config() -> SourceConfig:
    return SourceConfig(
        type="procedural",
        generators=list(_DEFAULT_RL_GENERATORS),
        seed=42,
        random_seed=False,
    )


@dataclass
class PrimeRLTrainerLossConfig:
    """Configures the Prime-RL trainer loss while tolerating submodule drift."""

    type: str = "default"
    # Current DPPO-style defaults.
    dppo_mask_low: float = 0.2
    dppo_mask_high: float = 0.2
    adv_tau: float = 1.0
    teacher_tau: float = 0.0
    kl_tau: float = 1e-3
    # Historical ratio-style defaults from older Prime-RL revisions.
    ratio_type: str = "token"
    token_mask_high: float = 8.0
    token_mask_low: float = 0.125
    sequence_clip_high: float = 10.0
    geo_mask_high: float = 10.0
    geo_mask_low: float = 0.1
    sequence_mask_low: float = 0.0
    sequence_mask_high: float = 100.0


@dataclass
class PrimeRLRLConfig:
    """Prime-RL training configuration."""

    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    init_adapter_path: Optional[str] = None

    output_dir: str = "outputs/prime_rl"
    clean: bool = False
    dump_subconfigs_only: bool = False

    # Checkpointing / resume (mirrors Prime-RL's SharedCheckpointConfig)
    ckpt_interval: Optional[int] = None
    ckpt_resume_step: Optional[int] = None  # -1 means "latest"
    ckpt_keep_last: Optional[int] = None
    ckpt_keep_interval: Optional[int] = None
    ckpt_save_adapter_separately: bool = True

    source: SourceConfig = field(default_factory=_default_procedural_source_config)
    scenario_backend: str = "remote_ssh"
    max_turns: int = 20

    batch_size: int = 32
    rollouts_per_example: int = 4
    learning_rate: float = 4.0e-5
    max_tokens: int = 2048
    seq_len: int = 4096
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: float = 64.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    )
    fused_lm_head_chunk_size: str = "disabled"
    trainer_impl: str = "auto"
    trainer_attn: str = "sdpa"
    activation_checkpoint_freq: int = 1
    inference_enable_auto_tool_choice: bool = False
    inference_tool_call_parser: Optional[str] = None
    inference_reasoning_parser: Optional[str] = None
    inference_gpu_memory_utilization: float = 0.3
    inference_enforce_eager: bool = False
    inference_max_model_len: int | None = None
    inference_gpu_ids: List[int] = field(default_factory=lambda: [0])
    trainer_gpu_ids: List[int] = field(default_factory=lambda: [0])
    max_steps: Optional[int] = None
    max_async_level: int = 1
    max_off_policy_steps: int = 8
    env_sampling_strategy: str = "round_robin"
    zero_advantage_filter_enforce: bool = False
    trainer_loss: PrimeRLTrainerLossConfig = field(
        default_factory=PrimeRLTrainerLossConfig
    )

    eval_on_benchmark: bool = False
    eval_every: int = 0
    eval_num_examples: int = -1
    eval_rollouts_per_example: int = 1
    eval_max_tokens: int | None = None

    wandb_project: str = "privesc-llm-prime-rl"
    wandb_run_name: Optional[str] = None


@dataclass
class RLConfig:
    """RL backend selection and backend-specific config."""

    backend: str = "prime_rl"
    prime_rl: PrimeRLRLConfig = field(default_factory=PrimeRLRLConfig)


CANONICAL_REWARD_MODES = (
    "outcome",
    "outcome_cost",
    "outcome_round",
    "outcome_round_cost",
)

DEPRECATED_REWARD_ALIASES = {
    "outcome_only": "outcome",
    "outcome_speed": "outcome_round",
    "outcome_speed_cost": "outcome_round_cost",
}

VALID_REWARD_MODES = (
    *CANONICAL_REWARD_MODES,
    *DEPRECATED_REWARD_ALIASES.keys(),
)


def normalize_reward_mode(mode: str) -> str:
    return DEPRECATED_REWARD_ALIASES.get(mode, mode)


@dataclass(frozen=True)
class PrivEscRewardConfig:
    mode: str = "outcome_round_cost"
    h_max: int = 20
    lambda_cost: float = 0.1
    c_ref_ms: float = 540_000.0
    llm_ms_clip_ms: float = 20_000.0
    tool_ms_clip_ms: float = 65_000.0
    iface_penalty: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", normalize_reward_mode(self.mode))
        if self.mode not in VALID_REWARD_MODES:
            valid = ", ".join(sorted(VALID_REWARD_MODES))
            raise ValueError(
                f"Invalid reward mode: {self.mode!r}. Expected one of: {valid}"
            )
        if self.h_max <= 0:
            raise ValueError("reward.h_max must be > 0")
        if self.c_ref_ms <= 0:
            raise ValueError("reward.c_ref_ms must be > 0")
        if self.lambda_cost < 0:
            raise ValueError("reward.lambda_cost must be >= 0")
        if self.llm_ms_clip_ms <= 0:
            raise ValueError("reward.llm_ms_clip_ms must be > 0")
        if self.tool_ms_clip_ms <= 0:
            raise ValueError("reward.tool_ms_clip_ms must be > 0")
        if self.iface_penalty < 0:
            raise ValueError("reward.iface_penalty must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "h_max": self.h_max,
            "lambda_cost": self.lambda_cost,
            "c_ref_ms": self.c_ref_ms,
            "llm_ms_clip_ms": self.llm_ms_clip_ms,
            "tool_ms_clip_ms": self.tool_ms_clip_ms,
            "iface_penalty": self.iface_penalty,
        }


RewardConfigLike: TypeAlias = PrivEscRewardConfig | Mapping[str, Any] | None


def resolve_reward_config(value: RewardConfigLike) -> PrivEscRewardConfig:
    if value is None:
        return PrivEscRewardConfig()
    if isinstance(value, PrivEscRewardConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("reward config must be a mapping or PrivEscRewardConfig")
    h_max = value.get("h_max")
    c_ref_ms = value.get("c_ref_ms")
    lambda_cost = value.get("lambda_cost")
    llm_ms_clip_ms = value.get("llm_ms_clip_ms")
    tool_ms_clip_ms = value.get("tool_ms_clip_ms")
    iface_penalty = value.get("iface_penalty")
    return PrivEscRewardConfig(
        mode=str(value.get("mode", "outcome_round_cost")),
        h_max=20 if h_max is None else int(h_max),
        lambda_cost=0.1 if lambda_cost is None else float(lambda_cost),
        c_ref_ms=540_000.0 if c_ref_ms is None else float(c_ref_ms),
        llm_ms_clip_ms=(20_000.0 if llm_ms_clip_ms is None else float(llm_ms_clip_ms)),
        tool_ms_clip_ms=(
            65_000.0 if tool_ms_clip_ms is None else float(tool_ms_clip_ms)
        ),
        iface_penalty=0.05 if iface_penalty is None else float(iface_penalty),
    )


@dataclass
class AppConfig:
    # Put _self_ first so file groups override any fallback values
    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"scenarios@scenario": "01_vuln_suid_gtfo"},
            {"ssh": "env"},
            {"agent": "openai"},
            {"runner": "eval_benchmark"},
            {"prompts": "evaluation"},
            {"datasets/sft": "collection/standard/training"},
            {"reasoning_variant_derivation": "no_reasoning"},
            {"datasets/rl": "privesc"},
            {"sft": "default"},
            {"sft/trl": "default"},
            {"sft/unsloth": "default"},
            {"rl": "default"},
            {"rl/prime_rl": "default"},
            {"reward": "default"},
            {"gtfobins": "default"},
            {"generators": "default"},
            # add experiments at runtime: +experiment=<name>
        ]
    )
    seed: int = 42
    log_level: str = "INFO"
    log_format: str = "%(asctime)s [%(name)s:%(levelname)s] %(message)s"
    paper_prompt_condition_slug: str = "prompt_detailed"
    ssh: SSHConfig = field(default_factory=SSHConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)
    reasoning_variant_derivation: SFTReasoningVariantDerivationConfig = field(
        default_factory=SFTReasoningVariantDerivationConfig
    )
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    reward: PrivEscRewardConfig = field(default_factory=PrivEscRewardConfig)
    gtfobins: GTFOBinsConfig = field(default_factory=GTFOBinsConfig)
    generators: GeneratorsConfig = field(default_factory=GeneratorsConfig)


def effective_max_assistant_turns(cfg: AppConfig) -> int:
    max_turns = int(cfg.agent.max_turns)
    if cfg.runner.mode == "trace_collection":
        return min(max_turns, int(cfg.datasets.sft.quality.max_turns))
    return max_turns


def sft_teacher_model(cfg: AppConfig) -> str:
    teacher_model = cfg.datasets.sft.model_name or cfg.datasets.sft.teacher_model
    if not teacher_model:
        raise ValueError("datasets.sft.teacher_model must be configured")
    return teacher_model


def register_with_hydra() -> None:
    cs = ConfigStore.instance()
    cs.store(name="app", node=AppConfig)


# Register at import time so other modules that import src.gym.config get the
# schema registered before Hydra tries to resolve defaults.
register_with_hydra()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def print_config(cfg: AppConfig) -> None:
    oc = OmegaConf.structured(cfg)
    logging.info(OmegaConf.to_yaml(oc, resolve=True))


if __name__ == "__main__":
    register_with_hydra()
    print_config()
