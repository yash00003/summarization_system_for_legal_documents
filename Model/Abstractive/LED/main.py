from transformers import LEDTokenizer, LEDForConditionalGeneration
import torch

def generate_led_summary(input_text, model_path="/Users/abhinavmittal/desktop/minor/model/abstractive/led", max_input_length=1024, max_output_length=256):
    if not hasattr(generate_led_summary, "tokenizer"):
        generate_led_summary.tokenizer = LEDTokenizer.from_pretrained(model_path)
        generate_led_summary.model = LEDForConditionalGeneration.from_pretrained(model_path)
        generate_led_summary.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generate_led_summary.model.to(generate_led_summary.device)

    try:
        inputs = generate_led_summary.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length
        )

        input_ids = inputs.input_ids.to(generate_led_summary.device)
        attention_mask = inputs.attention_mask.to(generate_led_summary.device)

        output_ids = generate_led_summary.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_output_length,
            num_beams=2,
            early_stopping=True
        )

        return generate_led_summary.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return ""