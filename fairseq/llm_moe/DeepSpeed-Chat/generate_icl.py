
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "training")))
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import deepspeed
import argparse
import torch


from moe_model.modeling_llama import LlamaForCausalLM, LlamaConfig
from moe_model.modeling_qwen import QWenLMHeadModel
from moe_model.configuration_qwen import QWenConfig
from moe_model.tokenization_qwen import QWenTokenizer
logger = logging.getLogger(__name__) 



def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_path",
        type=str,
        default="google/flan-t5-xl",
        help="Model id to use for inference.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="google/flan-t5-xl",
        help="tokenizer id to use for inference.",
    )
    parser.add_argument("--dataset_path", type=str, default="lm_dataset", help="Path to dataset.")
    parser.add_argument(
        "--src-lang",
        type=str,
        default="English",
        help="source language.",
    )
    parser.add_argument(
        "--tgt-lang",
        type=str,
        help="source language.",
    )
    parser.add_argument("--deepspeed_config",type=str,default="-")
    parser.add_argument(
		"--out_path",
		type=str,
		default='-',
		help="output path.",
	)
    parser.add_argument(
		"--local_rank",
		type=int,
	)
    parser.add_argument(
		"--is-moe",
		default=False,
        action='store_true'
	)
    parser.add_argument(
		"--eval-batch-size",
		default=1,
        type=int
	)

    
    args = parser.parse_args()
    return args


def inference_function(args):
    def read_text(path):
        new_list = []
        with open(path,'r') as r:
            for i in r.readlines():
                new_list.append(i.replace('\n',''))
        return new_list
    

    def get_prompt(src_lang, tgt_lang, q):
        if src_lang == "German":
            if tgt_lang == "English":
                return f"Translate the following sentence from German to English:\nGerman:In Pasco County wurden ebenfalls Algenblüten beobachtet.\nEnglish:Red Tide has also been observed in Pasco County.\nGerman:"+q+"\nEnglish:"
            else:
                raise ValueError(f"No prompt for {src_lang} to {tgt_lang}")
        elif src_lang == "English":
            if tgt_lang == 'Chinese':
                return f"Translate the following sentence from English to Chinese:\nEnglish:I have already stated I do not want to run for any local position.\nChinese:我已经说过我不想竞选任何本地职位。\nEnglish:"+q+"\nChinese:"
            if tgt_lang == 'German':
                return "Translate the following sentence from English to German:\nEnglish:Red Tide has also been observed in Pasco County.\nGerman:In Pasco County wurden ebenfalls Algenblüten beobachtet.\nEnglish:"+q+"\nGerman:"
            if tgt_lang == 'Romania':
                return "Translate the following sentence from English to Romania:\nEnglish:I have already stated I do not want to run for any local position.\nRomania:Am mai spus că nu vreau să candidez la nicio funcție locală.\nEnglish:"+q+"\nRomania:"
            if tgt_lang == 'French':
                return "Translate the following sentence from English to French:\nEnglish:from a long-term drive  to increase future freedom of action.\nFrench:d'une tendance sur le long terme à augmenter la liberté d'action future.\nEnglish:"+q+"\nFrench:"
            if tgt_lang == 'Arabic':
                return "Translate the following sentence from English to Arabic:\nEnglish:What are you scared of?\nArabic:مما أنت خائف؟ \nEnglish:"+q+"\nArabic:"
            if tgt_lang == 'Catalan':
                return "Translate the following sentence from English to Catalan:\nEnglish:You see Erso out there?\nCatalan:Veus a Erso?\nEnglish:"+q+"\nCatalan:"
            if tgt_lang == 'Spanish':
                return "Translate the following sentence from English to Spanish:\nEnglish:Humanitarian, recovery and development activities\nSpanish:Actividades humanitarias, de recuperación y de desarrollo\nEnglish:"+q+"\nSpanish:"
            if tgt_lang == 'Indonesia':
                return "Translate the following sentence from English to Indonesia:\nEnglish:Then perhaps I can be of assistance in a different way.\nIndonesia:Mungkin aku bisa menolongamu dengan cara lain.\nEnglish:"+q+"\nIndonesia:"
            if tgt_lang == 'Vietnamese':
                return "Translate the following sentence from English to Vietnamese:\nEnglish:Ostedes variegata is a species of beetle in the family Cerambycidae.\nVietnamese:Ostedes variegata là một loài bọ cánh cứng trong họ Cerambycidae.\nEnglish:"+q+"\nVietnamese:"
            if tgt_lang == 'Malayalam':
                return "Translate the following sentence from English to Malayalam:\nEnglish:So at 10:45 you go see Pirou at the hospital.\nMalayalam:അതിനാല്‍, 10.45ന് നീ ആശുപത്രിയില്‍ പോയി പിര്യൗവിനെ കാണണം.\nEnglish:"+q+"\nMalayalam:"
            if tgt_lang == 'French2':
                return "Translate the following sentence from English to French:\nEnglish:from a long-term drive  to increase future freedom of action.\nFrench:d'une tendance sur le long terme à augmenter la liberté d'action future.\nEnglish:More people watched the game because it was more exciting.\nFrench:Plus de personnes ont regardé les matchs parce que c'était plus excitant.\nEnglish:"+q+"\nFrench:"
            if tgt_lang == 'French4':
                return "Translate the following sentence from English to French:\nEnglish:from a long-term drive  to increase future freedom of action.\nFrench:d'une tendance sur le long terme à augmenter la liberté d'action future.\nEnglish:More people watched the game because it was more exciting.\nFrench:Plus de personnes ont regardé les matchs parce que c'était plus excitant.\nEnglish:How do we take that next step?\nFrench:Comment franchissons-nous l'étape suivante ?\nEnglish:Fans want to be on that field.French:Les fans veulent être sur ce terrain.\nEnglish:"+q+"\nFrench:"
            if tgt_lang == 'French6':
                return "Translate the following sentence from English to French:\nEnglish:from a long-term drive  to increase future freedom of action.\nFrench:d'une tendance sur le long terme à augmenter la liberté d'action future.\nEnglish:More people watched the game because it was more exciting.\nFrench:Plus de personnes ont regardé les matchs parce que c'était plus excitant.\nEnglish:How do we take that next step?\nFrench:Comment franchissons-nous l'étape suivante ?\nEnglish:Fans want to be on that field.French:Les fans veulent être sur ce terrain.\nEnglish:Does it predict artificial intelligence?\nFrench:Est-ce qu'elle prédit l'intelligence artificielle ?\nEnglish:We truly thank God for you.\nFrench:Nous remercions sincèrement Dieu de vous avoir.\nEnglish:"+q+"\nFrench:"
            if tgt_lang == 'French8':
                return "Translate the following sentence from English to French:\nEnglish:from a long-term drive  to increase future freedom of action.\nFrench:d'une tendance sur le long terme à augmenter la liberté d'action future.\nEnglish:More people watched the game because it was more exciting.\nFrench:Plus de personnes ont regardé les matchs parce que c'était plus excitant.\nEnglish:How do we take that next step?\nFrench:Comment franchissons-nous l'étape suivante ?\nEnglish:Fans want to be on that field.French:Les fans veulent être sur ce terrain.\nEnglish:Does it predict artificial intelligence?\nFrench:Est-ce qu'elle prédit l'intelligence artificielle ?\nEnglish:We truly thank God for you.\nFrench:Nous remercions sincèrement Dieu de vous avoir.\nEnglish:He never made it there.\nFrench:Il n'est jamais arrivé.\nEnglish:Just wanted to drop you a note.\nFrench:Je voulais seulement vous glisser un mot.\nEnglish:"+q+"\nFrench:"

                # return "Translate the following sentence from English to French:\nEnglish:from a long-term drive  to increase future freedom of action.\nFrench:d'une tendance sur le long terme à augmenter la liberté d'action future.\nThe translation of 'free' to 'liberté' is right.The translation of 'future' to 'liberté' is error.\nEnglish:"+q+"\nFrench:"

    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed()
    src_list = read_text(args.dataset_path)
    tgt_list = []
    tokenizer = QWenTokenizer.from_pretrained(args.tokenizer_path)


    if args.is_moe:
        from moe_model.configuration_qwen import QWenConfig
        from moe_model.modeling_qwen import QWenLMHeadModel


        config = QWenConfig(vocab_size=151936,
                    hidden_size=2048,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    emb_dropout_prob=0.1,
                    attn_dropout_prob=0.0,
                    layer_norm_epsilon=1e-5,
                    initializer_range=0.02,
                    max_position_embeddings=2048,
                    scale_attn_weights=True,
                    use_cache=True,
                    bf16=False,
                    fp16=False,
                    fp32=True,
                    kv_channels=128,
                    rotary_pct=1.0,
                    rotary_emb_base=10000,
                    use_dynamic_ntk=False,
                    use_logn_attn=False,
                    use_flash_attn=False,
                    intermediate_size=11008,
                    no_bias=True,
                    tie_word_embeddings=False,
                    seq_length=2048,
                    moe_expert_count=8,
                    sparse_step=4,
                    ep_size=2,
                    share_weight=0.9,
        )
        model = QWenLMHeadModel(config)
        state_dict = torch.load(f"{args.model_path}/mp_rank_00_model_states.pt", map_location="cpu")["module"]
        for layer_id in range(config.num_hidden_layers):
            if layer_id % config.sparse_step == 0:
                expert_num_per_gpu = config.moe_expert_count // config.ep_size
                for expert_id in range(expert_num_per_gpu*args.local_rank, expert_num_per_gpu*args.local_rank + expert_num_per_gpu):
                    moe_state_dict = torch.load(f"{args.model_path}/layer_{layer_id // config.sparse_step}_expert_{expert_id}_mp_rank_00_model_states.pt", map_location="cpu")
                    local_expert_id = expert_id - expert_num_per_gpu*args.local_rank
                    for key in moe_state_dict:
                        key_split = key.split('.')
                        key_split[-3] = str(local_expert_id)
                        new_key = '.'.join(key_split)
                        state_dict[new_key] = moe_state_dict[key]
                        print(f"Initializing {new_key} from {key}")
        model.load_state_dict(state_dict)
    else:
        from moe_model.modeling_qwen import QWenLMHeadModel
        model = QWenLMHeadModel.from_pretrained(args.model_path, torch_dtype="auto", device_map=args.local_rank)
    model.eval() # !
    model = deepspeed.init_inference(model=model, config=args.deepspeed_config)

    logger.info(model)
    max_new_tokens = 512
    batch_size = args.eval_batch_size
    batch = []
    f_out = open(args.out_path, 'w')

    generation_kwargs = {"early_stopping": True}
    # 当pad和eos相同时，transformers默认不生成pad的attention mask，通过is_pad_token_not_equal_to_eos_token_id，这里做了修改 transformers/generation/utils.py 609行
    # qwen的attention代码中仅针对casual mask做了掩码，没有对pad的attention进行掩码，这里做了修改 modeling_qwen.py 255行
    pad_token_id = None
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    elif hasattr(tokenizer, "eod_id") and tokenizer.eod_id is not None:
        pad_token_id = tokenizer.eod_id
    if pad_token_id:
        generation_kwargs["pad_token_id"] = pad_token_id

    eos_token_id = None
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        eos_token_id = tokenizer.eos_token_id
    elif hasattr(tokenizer, "eod_id") and tokenizer.eod_id is not None:
        eos_token_id = tokenizer.eod_id
    if eos_token_id:
        generation_kwargs["eos_token_id"] = eos_token_id

    print(f"generation kwargs: {generation_kwargs}")
    for item_idx, item in enumerate(src_list):
        batch.append(item.strip())
        if len(batch) == batch_size or item_idx == len(src_list) - 1:
            # q = "For readers outside of Wales In Welsh twp means daft and pwp means poo. Translate this sentence into English language. "
            # q = "Translate the following sentence from English to German:\nEnglish:Red Tide has also been observed in Pasco County.\nGerman:In Pasco County wurden ebenfalls Algenblüten beobachtet.\nEnglish:"+item+"\nGerman:"
            batch_input_s = [get_prompt(args.src_lang, args.tgt_lang, s) for s in batch]
            inputs = [tokenizer.encode(input_s) for input_s in batch_input_s ]
            max_length = max([len(inp) for inp in inputs])

            padded_inputs = []
            for inp in inputs:
                padded_input = [pad_token_id] * (max_length - len(inp)) + inp
                if max_length - len(inp) > 0:
                    assert pad_token_id is not None, "No pad token in tokenizer, please set eval-batch-size=1"
                padded_inputs.append(padded_input)
            
            
            inputs = torch.LongTensor(padded_inputs).to(args.local_rank)
            # left pad
            
            outputs = model.generate(inputs, max_new_tokens=max_new_tokens, **generation_kwargs)


            outputs_str = tokenizer.batch_decode(outputs)
            prefix_str = tokenizer.batch_decode(inputs)
            for str_idx, out_str in enumerate(outputs_str):
                prefix = prefix_str[str_idx]
                result = out_str[len(prefix):].split('\n')[0].replace('<|endoftext|>', '').replace('</s>', '').replace('\n', '').replace('<pad>', '')
                print_str = out_str.replace('</s>', '').replace('<pad>', '')
                print(f"idx: {item_idx-len(batch)+1+str_idx}, rank: {args.local_rank}\nOUTPUT: {print_str}\nRESULT: {result}")
                tgt_list.append(result)
            
            if args.local_rank == 0:
                for result in tgt_list:
                    f_out.write(result.strip() + '\n')

            batch, tgt_list = [], []

    f_out.close()


def main():
    args = parse_args()
    inference_function(args)


if __name__ == "__main__":
    main()