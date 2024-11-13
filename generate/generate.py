import os
import argparse
import json

import torch

from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import base64
import io
import re
import random


from transformers import AutoTokenizer, AutoModel

# cotprompt= '''
# Please solve the Relation Extraction task. The answer relation must be selected from [race, place_of_birth, present_in, alternate_names, parent, locate_at, neighbor, part_of, alumni, siblings, religion, place_of_residence, nationality, contain, couple, awarded, subsidiary, peer, held_on, member_of, charges, None].
# Given the context and image, figure out the reasoning steps that lead to the relation between two entities to be the specific one .
# Context: {}
# HEAD_ENTITY:{}
# TAIL_ENTITY:{}

# Given the context, what's the relation between [HEAD_ENTITY] and [TAIL_ENTITY]?
# the answer relation must be selected from [race, place_of_birth, present_in, alternate_names, parent, locate_at, neighbor, part_of, alumni, siblings, religion, place_of_residence, nationality, contain, couple, awarded, subsidiary, peer, held_on, member_of, charges, None], known the relation is [relation], the reasoning steps are:
# [the reason and evidence of the answer]
# So the relation between [HEAD_ENTITY] and [TAIL_ENTITY] is [relation]
# '''

cotprompt= '''
Please solve the Relation Extraction task. The answer relation must be selected from [race, place_of_birth, present_in, alternate_names, parent, locate_at, neighbor, part_of, alumni, siblings, religion, place_of_residence, nationality, contain, couple, awarded, subsidiary, peer, held_on, member_of, charges, None].
Given the context and image, figure out the reasoning steps that lead to the relation between two entities to be the specific one .
Context: {}
HEAD_ENTITY:{}
TAIL_ENTITY:{}

Given the context and image, what's the relation between [HEAD_ENTITY] and [TAIL_ENTITY] and give the reason?
the answer relation must be selected from the given relation. Known the relation is [relation], the reasoning steps are:
[the reason and evidence of the answer]
So the relation between [HEAD_ENTITY] and [TAIL_ENTITY] is [relation]
'''

prompt = '''
Head: {}
Tail: {}

The known relationships are [race, place_of_birth, present_in, alternate_names, parent, locate_at, neighbor, part_of, alumni, siblings, religion, place_of_residence, nationality, contain, couple, awarded, subsidiary, peer, held_on, member_of, charges, None]. 

Given the input image, 'head', and 'tail', identify the appropriate relationship between 'head' and 'tail' from the above list. The 'head' and 'tail' are provided as input entities. 

Select the relationship that best describes the connection between 'head' and 'tail' based on the provided image and output the result in JSON format with 'relation', 'head', and 'tail'. 

For example:
Output: ['relation': '', 'head': '', 'tail': '']
 '''
info_dict = {
    "race": 1,
    "place_of_birth": 2,
    "present_in": 3,
    "alternate_names": 4,
    "parent": 5,
    "locate_at": 6,
    "neighbor": 7,
    "part_of": 8,
    "alumni": 9,
    "siblings": 10,
    "religion": 11,
    "place_of_residence": 12,
    "nationality": 13,
    "contain": 14,
    "couple": 15,
    "awarded": 16,
    "subsidiary": 17,
    "peer": 18,
    "held_on": 19,
    "member_of": 20,
    "charges": 21,
    "None": 22  # 特殊值None，没有对应数字
}
def get_answer(outputs):
    outputs = outputs.replace('\\', '')

    for key in info_dict:
        if re.search(key, outputs, re.IGNORECASE):
            return str(info_dict[key])  # 返回匹配键对应的值
    return "22"

def minigpt4_finetune_parser():
    parser = argparse.ArgumentParser(description="finetune minigpt4")
    parser.add_argument("--cfg-path", default="configs/minigpt4_infer_fp16.yaml", help="path to configuration file.")
    parser.add_argument("--name", type=str, default="A2", help="evaluation name")
    parser.add_argument("--ckpt", type=str, help="path to configuration file.")
    parser.add_argument("--eval_opt", type=str, default="all", help="path to configuration file.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument('--output_path', default='/home/nfs02/xingsy/code/output/llava1.5.txt')
    parser.add_argument('--file_path', default='/home/nfs02/xingsy/code/data/MNRE/mnre_txt/mnre_test.txt')
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of GPU to use')
    parser.add_argument('--random_flag', type=int, default=0, help='Index of GPU to use')
    return parser

class Minigpt:
    def __init__(self, model_path) -> None:
        from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser

        self.model, self.processor = init_model(minigpt4_finetune_parser().parse_args())

    def chat(self, input):
        from minigpt4.conversation.conversation import CONV_VISION_minigptv2
        from minigpt4.common.eval_utils import prepare_texts
        CONV_VISION = CONV_VISION_minigptv2
        conv_temp = CONV_VISION.copy()
        msgs = input['question']
        image = input["image"]
        raw_image = Image.open(image_path).convert('RGB')
        images = self.processor(raw_image).unsqueeze(0)
        text_chunks = [msgs]  # 在这个例子中，假设 msgs 是一个长文本，直接作为一个片段传递
        texts = prepare_texts(text_chunks, CONV_VISION)


        outputs = self.model.generate(images, texts, max_new_tokens=200, num_beams=1, temperature=0.1)[0]

        print(outputs)
        return outputs

class Owlplug:
    def __init__(self, model_path) -> None:
        import torch
        from PIL import Image
        from transformers import TextStreamer

        from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from mplug_owl2.conversation import conv_templates, SeparatorStyle
        from mplug_owl2.model.builder import load_pretrained_model
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")



    def chat(self, input):
        import torch
        from PIL import Image
        from transformers import TextStreamer

        from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from mplug_owl2.conversation import conv_templates, SeparatorStyle
        from mplug_owl2.model.builder import load_pretrained_model
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        image_file = input["image"] # Image Path
        query = input["question"]
        conv =  conv_templates["mplug_owl2"].copy()
        roles = conv.roles

        image = Image.open(image_file).convert('RGB')
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))

        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        inp = DEFAULT_IMAGE_TOKEN + query
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        temperature = 0.7
        max_new_tokens = 512

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                num_beams=1,
                stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        print(outputs)

        return outputs


class BlipModel:
    def __init__(self, model_path):
        """Initialize BLIP model and processor."""
        import torch
        import requests
        from PIL import Image
        from transformers import Blip2Processor, Blip2ForConditionalGeneration


        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float32)
        self.model.to("cuda")
        # from lavis.models import load_model_and_preprocess
        # self.model, self.processor, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", 
        #                                                             model_type="vicuna7b", 
        #                                                             is_eval=True, 
        #                                                             device=device)

    def chat(self, input):
        """Process input and return the model's prediction."""
        question = input['question']
        image_path = input['image']
        # raw_image = Image.open(image_path).convert('RGB')
        # image_ = self.processor["eval"](raw_image).unsqueeze(0).to(device)
        # caption_test = self.model.generate({"image": image_, "prompt": question})
        # print(caption_test)
        # Load and process image
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.processor(raw_image, question, return_tensors="pt").to("cuda", torch.float32)
        
        out = self.model.generate(**inputs,
                max_length=200,
                min_length=1, 
                temperature=1.0,
                num_beams=5,
                max_new_tokens=200,
                length_penalty=1.5,
                repetition_penalty=2.0,
                use_nucleus_sampling=True)

        # Decode and retrieve text
        result = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        # 使用正则表达式提取结果中的数字
        print(result)
        return get_answer(result)


class LlavaModel:
    def __init__(self, model_path) -> None:
        model_name='llava-v1.5-7b'
        tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base=None,model_name=model_name)
        self.tokenizer=tokenizer
        self.model=model
        self.image_processor=image_processor
        self.context_len=context_len

    def chat(self, input):
        msgs = input['question']
        if self.model.config.mm_use_im_start_end:
            msgs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + msgs
        else:
            msgs = DEFAULT_IMAGE_TOKEN + '\n' + msgs

        image = Image.open(input['image']).convert('RGB')
        conv = conv_templates["llava_v1"].copy()
        conv.append_message(conv.roles[0], msgs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True,
                temperature=1.0,
                num_beams=2,
                max_new_tokens=1024,
                length_penalty=1.5,
                use_cache=True)
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                # Use regex to extract the numeric value from outputs
        outputs = outputs.lower()
        print(outputs)
        return outputs



class Aligngpt:

    def __init__(self, model_path) -> None:
        import sys
        sys.path.append('/home/nfs02/xingsy/code/AlignGPT')
        from AlignGPT.src.utils.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, INFERENCE
        from AlignGPT.src.utils.conversation import conv_templates, SeparatorStyle
        from AlignGPT.src.model.builder import load_pretrained_model
        from AlignGPT.src.utils.general_utils import disable_torch_init
        from AlignGPT.src.utils.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, None, "aligngpt-7b")
        self.model.set_stage(INFERENCE)

    def chat(self, input):
        msgs = input['question']
        # TODO: check version

        conv_mode = "llava_v0"


        conv = conv_templates[conv_mode].copy()

        roles = conv.roles

        # image = Image.open(input['image']).convert('RGB')
        # # Similar operation in model_worker.py
        # image_tensor = process_images([image], self.image_processor, self.model.config)
        # if type(image_tensor) is list:
        #     image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        # else:
        #     image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        image = None
        inp = msgs
        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            
            conv.append_message(conv.roles[0], inp)
        image_tensor = None
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        #print(prompt)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        from transformers import TextStreamer
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False ,
                temperature=0.0,
                max_new_tokens=200,
                use_cache=True
               )

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        # print(outputs)
        return get_answer(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='/home/nfs02/xingsy/code/output/temp.txt')
    parser.add_argument('--file_path', default='/home/nfs02/xingsy/code/data/MNRE/mnre_txt/mnre_test.txt')
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of GPU to use')
    parser.add_argument('--random_flag', type=int, default=0, help='Index of GPU to use')
    args = parser.parse_args()

    random_flag = 0
    device = torch.device(f'cuda:{args.gpu_index}' if torch.cuda.is_available() else 'cpu')
    #chat_model = LlavaModel('/home/nfs03/xingsy/playground/model/llava-v1.5-7b')
    #chat_model = Owlplug("/home/nfs02/xingsy/code/checkpoint")  
    # chat_model = BlipModel("/home/nfs02/xingsy/code/blip2")
    #chat_model = Minigpt('')
    chat_model = Aligngpt('/home/nfs02/xingsy/code/AlignGPT/playground/model/aligngpt-7b')
    output_file = args.output_path
    file_path = args.file_path
    with open(output_file, "a+") as f:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc="Processing", unit=" lines"):  # 使用 tqdm 包装文件迭代
                # 解析每一行的JSON数据
                data = eval(line.strip())
                # 处理文本数据
                sentence = ' '.join(data['token'])
                head = data['h']['name']
                tail = data['t']['name']

                text = prompt.format(head, tail)
                # 处理图像数据（示例中仅打开图像，实际应用中可能需要进行预处理）
                image_path = '/home/nfs02/xingsy/code/data/MNRE/mnre_image/img_org/test/' + data['img_id']
                image_filename = ''
                if random_flag == 1:
                    image_dir = '/home/nfs02/xingsy/code/data/MNRE/mnre_image/img_org/train/'
                    image_filename = random.choice(os.listdir(image_dir))
                    image_path = os.path.join(image_dir, image_filename)
                    inputs = {"image": image_path, "question": text}
                    answer = chat_model.chat(inputs)
                    outputs = answer.replace("\n", "")
                    result = " ### ".join([data['img_id'], image_filename, outputs])
                    print("random", result)
                    f.write(result + "\n")
                else:
                    inputs = {"image": image_path, "question":text}
                    answer = chat_model.chat(inputs)
                    outputs = answer.replace("\n", "")
                    result = " ### ".join([data['img_id'], data['relation'], outputs])
                    # print("sentence", result)
                    # f.write(result + "\n")