from traceback import format_exc

import hivemind
from flask import jsonify, request

import config
from app import app, models
from utils import safe_decode

logger = hivemind.get_logger(__file__)


@app.post("/api/v1/generate")
def http_api_generate():
    try:
        model_name = get_typed_arg("model", str, config.DEFAULT_MODEL_NAME)
        inputs = request.values.get("inputs")
        do_sample = get_typed_arg("do_sample", int, False)
        temperature = get_typed_arg("temperature", float)
        top_k = get_typed_arg("top_k", int)
        top_p = get_typed_arg("top_p", float)
        max_length = get_typed_arg("max_length", int)
        max_new_tokens = get_typed_arg("max_new_tokens", int)
        session_id = request.values.get("session_id")
        logger.info(f"generate(), model={repr(model_name)}, inputs={repr(inputs)}")

        if session_id is not None:
            raise RuntimeError(
                "Reusing inference sessions was removed from HTTP API, please use WebSocket API instead"
            )

        model, tokenizer = models[model_name]

        if inputs is not None:
            inputs = tokenizer(inputs, return_tensors="pt")["input_ids"].to(config.DEVICE)
            n_input_tokens = inputs.shape[1]
        else:
            n_input_tokens = 0

        outputs = model.generate(
            inputs=inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        outputs = safe_decode(tokenizer, outputs[0, n_input_tokens:])
        logger.info(f"generate(), outputs={repr(outputs)}")

        return jsonify(ok=True, outputs=outputs)
    except Exception:
        return jsonify(ok=False, traceback=format_exc())

# Create a chat_completions endpoint with the required logic to be call compatible with OpenAI style api call
@app.post("/api/v3/generate")
def chat_completions():
    try:
        # Extract parameters from the request
        model_name = request.json.get("model", config.DEFAULT_MODEL_NAME)
        messages = request.json.get("messages", [])
        temperature = request.json.get("temperature", 1.0)
        top_p = request.json.get("top_p", 1.0)
        n = request.json.get("n", 1)
        max_tokens = request.json.get("max_tokens", None)
        # TODO: Handle other parameters as required

        # Check if the model with the specified name exists
        if model_name not in models:
            return jsonify(ok=False, error=f"Model {model_name} not found"), 400

        model, tokenizer = models[model_name]

        # Tokenize the messages and generate response
        # For simplicity, we'll concatenate the messages and generate a response.
        # A more sophisticated approach could involve handling each message individually.
        concatenated_messages = " ".join([msg['content'] for msg in messages])
        inputs = tokenizer(concatenated_messages, return_tensors="pt")["input_ids"].to(config.DEVICE)
        outputs = model.generate(
            inputs=inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_length=max_tokens
        )
        generated_response = safe_decode(tokenizer, outputs[0])

        # Format the response
        response = {
            "id": "chatcmpl-" + str(hash(generated_response)),  # A mock ID for demonstration
            "object": "chat.completion",
            "created": int(time.time()),  # Current timestamp
            "model": model_name,
            "usage": {"prompt_tokens": len(inputs[0]), "completion_tokens": len(outputs[0]), "total_tokens": len(inputs[0]) + len(outputs[0])},
            "choices": [{"message": {"role": "assistant", "content": generated_response}, "finish_reason": "stop"}]
        }

        return jsonify(response)

    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500
    

def get_typed_arg(name, expected_type, default=None):
    value = request.values.get(name)
    return expected_type(value) if value is not None else default
