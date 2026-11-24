from vllm import LLM, SamplingParams

def main():
    llm = LLM(
        model="unsloth/Llama-3.2-1B",
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True
    )

    params = SamplingParams(temperature=0.0, max_tokens=32)

    out = llm.generate(["The capital of France is"], params)
    
    print(out[0].outputs[0].text)

if __name__ == "__main__":
    main()