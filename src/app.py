import torch
import importlib
from transformers import AutoProcessor
from huggingface_hub import create_repo, upload_large_folder, login
import os
import shutil
import gradio as gr

from typing import Iterable
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red",
    c50="#FFF0E5",
    c100="#FFE0CC",
    c200="#FFC299",
    c300="#FFA366",
    c400="#FF8533",
    c500="#FF4500",
    c600="#E63E00",
    c700="#CC3700",
    c800="#B33000",
    c900="#992900",
    c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

SUPPORTED_ARCHITECTURES = {
    "Qwen3_5ForConditionalGeneration": {
        "module": "transformers",
        "class_name": "Qwen3_5ForConditionalGeneration",
        "label": "Qwen 3.5 (Multimodal / Conditional Generation)",
    },
    "Qwen3VLForConditionalGeneration": {
        "module": "transformers",
        "class_name": "Qwen3VLForConditionalGeneration",
        "label": "Qwen 3 VL (Vision-Language)",
    },
    "Qwen2_5_VLForConditionalGeneration": {
        "module": "transformers",
        "class_name": "Qwen2_5_VLForConditionalGeneration",
        "label": "Qwen 2.5 VL (Vision-Language)",
    },
    "Qwen2VLForConditionalGeneration": {
        "module": "transformers",
        "class_name": "Qwen2VLForConditionalGeneration",
        "label": "Qwen 2 VL (Vision-Language)",
    },
}

def get_model_class(architecture: str):
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture '{architecture}'")
    info = SUPPORTED_ARCHITECTURES[architecture]
    module = importlib.import_module(info["module"])
    cls = getattr(module, info["class_name"], None)
    if cls is None:
        raise ImportError(f"Cannot find {info['class_name']} in {info['module']}.")
    return cls

def load_and_reupload_model(model_name, new_repo_id, hf_token, max_shard_size, selected_arch_label):

    log_output = []
    local_dir = None

    try:
        if not model_name or not new_repo_id or not hf_token:
            return "Error: Model name, repo id, and token are required."

        arch_options = list(SUPPORTED_ARCHITECTURES.keys())
        arch_labels = [SUPPORTED_ARCHITECTURES[a]["label"] for a in arch_options]
        selected_arch = arch_options[arch_labels.index(selected_arch_label)]

        login(token=hf_token)
        log_output.append("Hugging Face login successful")

        create_repo(
            repo_id=new_repo_id,
            private=True,
            exist_ok=True
        )

        log_output.append(f"Repo ready: {new_repo_id}")

        log_output.append(f"Loading processor: {model_name}")

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        log_output.append("Processor loaded")

        log_output.append(f"Loading model: {model_name} [{selected_arch}]")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        ModelClass = get_model_class(selected_arch)

        model = ModelClass.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            use_safetensors=True
        )

        model.eval()

        log_output.append(f"Model loaded on {device}")

        local_dir = f"_resharded_{os.urandom(4).hex()}"

        os.makedirs(local_dir, exist_ok=True)

        log_output.append(
            f"Saving model shards (max_shard_size={max_shard_size})"
        )

        model.save_pretrained(
            local_dir,
            max_shard_size=max_shard_size
        )

        processor.save_pretrained(local_dir)

        log_output.append("Model and processor saved locally")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        log_output.append("Uploading model to HuggingFace...")

        upload_large_folder(
            repo_id=new_repo_id,
            repo_type="model",
            folder_path=local_dir,
            revision="main"
        )

        log_output.append("Upload completed successfully!")
        log_output.append(f"Model live at https://huggingface.co/{new_repo_id}")

    except Exception as e:
        log_output.append(f"Error: {str(e)}")

    finally:
        if local_dir and os.path.exists(local_dir):
            try:
                shutil.rmtree(local_dir)
                log_output.append("Temporary files cleaned up")
            except Exception as ce:
                log_output.append(f"Cleanup warning: {ce}")

    return "\n".join(log_output)


with gr.Blocks() as demo:

    gr.Markdown(
        """
# Model Sharder & Re-Uploader

This tool will:

1. Download a large Vision-Language or Text model  
2. Save it locally with smaller safetensor shards  
3. Upload it to a private Hugging Face repository

Uses upload_large_folder() for reliable large uploads.
"""
    )
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = round(torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 2)
        gr.Markdown(f"**Hardware Status:** GPU Available: **{gpu_name}** ({gpu_mem} GB)")
    else:
        gr.Markdown("**Hardware Status:** Running on **CPU only**. Processing may be very slow.")

    with gr.Row():

        with gr.Column(scale=2):

            model_name = gr.Textbox(
                label="Original Model Name",
                value="Qwen/Qwen3-VL-2B-Instruct"
            )

            new_repo_id = gr.Textbox(
                label="New Repository ID",
                placeholder="username/my-private-resharded-model"
            )

            hf_token = gr.Textbox(
                label="HuggingFace Write Token",
                type="password",
                placeholder="hf_xxxxxxxxx"
            )

            max_shard_size = gr.Textbox(
                label="Max Shard Size",
                value="4.4GB"
            )
            
            arch_options = list(SUPPORTED_ARCHITECTURES.keys())
            arch_labels = [SUPPORTED_ARCHITECTURES[a]["label"] for a in arch_options]
            
            selected_arch_label = gr.Dropdown(
                choices=arch_labels,
                label="Model Architecture",
                value=arch_labels[0]
            )

            run_btn = gr.Button(
                "Shard & Upload Model",
                variant="primary"
            )

        with gr.Column(scale=3):

            logs = gr.Textbox(
                label="Process Logs",
                lines=20,
                interactive=False,
                autoscroll=True
            )

    run_btn.click(
        fn=load_and_reupload_model,
        inputs=[model_name, new_repo_id, hf_token, max_shard_size, selected_arch_label],
        outputs=logs
    )

    gr.Markdown(
        """
        ### Supported Architectures
        - Qwen 3.5 (`Qwen3_5ForConditionalGeneration`)
        - Qwen 3 VL (`Qwen3VLForConditionalGeneration`)
        - Qwen 2.5 VL (`Qwen2_5_VLForConditionalGeneration`)
        - Qwen 2 VL (`Qwen2VLForConditionalGeneration`)
        """
    )


if __name__ == "__main__":
    demo.launch(theme=orange_red_theme, mcp_server=True, ssr_mode=False, show_error=True, share=True)
