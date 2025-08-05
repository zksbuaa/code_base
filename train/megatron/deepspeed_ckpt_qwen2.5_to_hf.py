import torch
import transformers
import glob
import argparse
import json
import os
import sys
megatron_path = "/mnt/public2/code/ssc/dynamic_train/Megatron-DeepSpeed-dynamic"
sys.path.append(megatron_path)

megatron_deepspeed_model_path = '/mnt/public2/code/ssc/dynamic_train/expt/0415-2-3b-example/ckpt/global_step22872/'


parser = argparse.ArgumentParser(description='')
parser.add_argument('--input', type=str, default=megatron_deepspeed_model_path)
parser.add_argument('--size', type=str, default='3B', choices=['0.5B', '32B', '7B', '72B'], help='32B, 7B or 72B')
parser.add_argument('--output', type=str, default='')
parser.add_argument('--vocab_size', type=int, default=151936)  # 与训练时一致

args = parser.parse_args()

if not args.input.endswith('/'):
    args.input += '/'

if args.output == "":
    save_path = args.input.rstrip('/') + "_hf"
    print("No output dir set. Setting output dir as ", save_path)
else:
    save_path = args.output

if args.size == '3B':
    hf_model_path =  "/mnt/public/model/zks/Qwen2.5-3B/"
else:
    print("Error model size choice, exitting...")
    exit(0)


vocab_size = args.vocab_size
ckpt_path = args.input
layers = glob.glob(ckpt_path + 'mp*')

qwen_config = transformers.AutoConfig.from_pretrained(hf_model_path)
if int(qwen_config.num_key_value_heads) < int(qwen_config.num_attention_heads):
    use_gqa = True 
else:
    use_gqa = False


chunk_size = int(qwen_config.hidden_size / qwen_config.num_key_value_heads)   # 1024
width_per_head = int(qwen_config.hidden_size / qwen_config.num_attention_heads)  # 128
q_kv_ratio = int(qwen_config.num_attention_heads / qwen_config.num_key_value_heads)   # 8

model = transformers.AutoModelForCausalLM.from_pretrained(hf_model_path)
model.model.embed_tokens = torch.nn.Embedding(vocab_size, qwen_config.hidden_size)
model.lm_head = torch.nn.Linear(qwen_config.hidden_size, vocab_size, bias=False)

#last_layer = int(max([layer.split('/')[-1].split("_")[1].split('-')[0] for layer in layers]))
max_tp_rank = 1 #int(max([layer.split('/')[-1].split("_")[2].split('-')[0] for layer in layers]))

model_keys = model.state_dict().keys()
new_dict = {k: [] for k in model_keys}

model.config.vocab_size = vocab_size
model.config.torch_dtype = "bfloat16"

if use_gqa:
    for tp_rank in range(max_tp_rank + 1):
        weights = f"{ckpt_path}mp_rank_{tp_rank:02d}_model_states.pt"
        state_dict = torch.load(weights, map_location='cpu')
        if tp_rank == 0:
            new_dict['model.embed_tokens.weight'].append(state_dict['module']['language_model']['embedding']['word_embeddings']['weight'])
            new_dict['lm_head.weight'].append(state_dict['module']['language_model']['output_layer']['weight']) 
            new_dict['model.norm.weight'].append(state_dict['module']['language_model']['encoder']['final_layernorm.weight'])
        for num_layer in range(36):
            if tp_rank == 0:
                new_dict[f"model.layers.{num_layer}.input_layernorm.weight"].append(state_dict['module']['language_model']['encoder'][f'layers.{num_layer}.input_layernorm.weight'])
                new_dict[f"model.layers.{num_layer}.post_attention_layernorm.weight"].append(state_dict['module']['language_model']['encoder'][f'layers.{num_layer}.post_attention_layernorm.weight'])
            q_width, kv_width = qwen_config.hidden_size // qwen_config.num_key_value_heads, width_per_head   ## 1024, 128
            qkv_width = q_width + 2 * kv_width   # 1280
            new_dict[f"model.layers.{num_layer}.self_attn.q_proj.weight"].append(state_dict['module']['language_model']['encoder'][f'layers.{num_layer}.self_attention.query_key_value.weight'][:q_width, :])
            new_dict[f"model.layers.{num_layer}.self_attn.k_proj.weight"].append(state_dict['module']['language_model']['encoder'][f'layers.{num_layer}.self_attention.query_key_value.weight'][q_width:q_width + kv_width, :])
            new_dict[f"model.layers.{num_layer}.self_attn.v_proj.weight"].append(state_dict['module']['language_model']['encoder'][f'layers.{num_layer}.self_attention.query_key_value.weight'][q_width + kv_width:q_width + 2 * kv_width, :])
            new_dict[f"model.layers.{num_layer}.self_attn.q_proj.bias"].append(state_dict['module']['language_model']['encoder'][f'layers.{num_layer}.self_attention.query_key_value.bias'][:q_width])
            new_dict[f"model.layers.{num_layer}.self_attn.k_proj.bias"].append(state_dict['module']['language_model']['encoder'][f'layers.{num_layer}.self_attention.query_key_value.bias'][q_width:q_width + kv_width])
            new_dict[f"model.layers.{num_layer}.self_attn.v_proj.bias"].append(state_dict['module']['language_model']['encoder'][f'layers.{num_layer}.self_attention.query_key_value.bias'][q_width + kv_width:q_width + 2 * kv_width])
            # attention dense
            new_dict[f"model.layers.{num_layer}.self_attn.o_proj.weight"].append(state_dict['module']['language_model']['encoder'][f'layers.{num_layer}.self_attention.dense.weight'])
            half_dense_size = int(state_dict['module']['language_model']['encoder'][f'layers.{num_layer}.mlp.dense_h_to_4h.weight'].shape[0] / 2)
            new_dict[f"model.layers.{num_layer}.mlp.gate_proj.weight"].append(state_dict['module']['language_model']['encoder'][f'layers.{num_layer}.mlp.dense_h_to_4h.weight'][:half_dense_size, :])
            new_dict[f"model.layers.{num_layer}.mlp.up_proj.weight"].append(state_dict['module']['language_model']['encoder'][f'layers.{num_layer}.mlp.dense_h_to_4h.weight'][half_dense_size:, :])
            new_dict[f"model.layers.{num_layer}.mlp.down_proj.weight"].append(state_dict['module']['language_model']['encoder'][f'layers.{num_layer}.mlp.dense_4h_to_h.weight'])

    for k in model_keys:
        if "o_proj" in k or "down_proj" in k:
            new_dict[k] = torch.cat(new_dict[k], dim=1)
        else:
            new_dict[k] = torch.cat(new_dict[k])

    #print(f'saving model to {save_path}')
    #print(new_dict.keys())
    model.load_state_dict(new_dict)
    #print(model)
    #model.save_pretrained(save_path, state_dict = new_dict, from_pt=True)
    #print('save done, testing..')


# 测试
#model = transformers.AutoModelForCausalLM.from_pretrained(save_path)
model.to('cuda')
tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_path, padding_side="left")



input_text = ["In the morning, I", "我今天", "我的名字", "My name is"]

# 将输入文本转换为模型输入
inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)

# 生成文本
outputs = model.generate(**inputs, max_new_tokens=20)

output_text_list = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

print(output_text_list)

test_zh = [
    '在那个阳光明媚、微风轻拂的早晨，小明决定离开他那位于高楼林立的城市中心的小公寓，前往绿树成荫的公园散步。他想要在这个快节奏的生活环境中寻找一片宁静之地，远离城市的喧嚣与繁忙。公园里的空气清新而芬芳，伴随着鸟儿欢快的歌声和孩子们无邪的笑声，仿佛时间在此刻凝固，让人感受到一种久违的放松和平静。随着步伐的移动，他看到了一位老人在练太极，动作优雅而缓慢，每一招每一式都充满了力量与和谐，这使他想起了自己小时候跟爷爷学习武术的日子。',
    '她是一个热爱生活、热爱音乐的女孩。每当她感到疲惫或者心情低落的时候，她总是会拿起那把陪伴她多年的钢琴，开始弹奏起那些她最喜欢的曲子。音符在她的指尖跳跃，她的心情也随之变得愉悦起来。她喜欢那种用音乐来表达情感的方式，因为在那一刻，她可以尽情地释放自己的情感，让自己的心灵得到一种解脱。她相信音乐是一种治愈心灵的力量，它可以让人感受到一种美好的情感，让人感到幸福与满足。',
    '他们是一群年轻人，他们相识于大学时代，彼此之间有着深厚的友谊。他们一起度过了许多美好的时光，一起经历了许多难忘的瞬间。他们曾一起去海边旅行，一起在山上露营，一起在城市里逛街。他们之间没有任何隔阂，没有任何芥蒂，只有纯真的友谊和深深的情感。他们相信友谊是一种宝贵的财富，是一种无法取代的情感，是一种永远不会消逝的记忆。',
    '桌子上的那本书是一本关于自然界的书，它讲述了大自然的奥秘和神奇。书中的插图栩栩如生，文字简洁明了，让人一目了然。书中的内容涉及了植物、动物、天文等方方面面，让人感受到了大自然的伟大和神秘。书中还有许多有趣的故事和趣味的知识，让人在阅读的过程中不仅能够获得知识，还能够获得乐趣。这本书是一本值得一读的好书，它可以让人感受到大自然的美丽和奇妙。',
    '信息技术的发展改变了人们的沟通方式，社交媒体平台让人们即使相隔万里也能保持密切联系。朋友之间可以通过即时通讯工具随时交流最新的动态，分享喜怒哀乐；家人也可以利用视频通话功能，即使身处异国他乡也仿佛近在咫尺。然而，这种便捷性也带来了一些挑战，比如隐私问题和个人信息安全。因此，在享受网络带来的便利时，我们也应该增强自我保护意识，合理使用互联网资源。',
    '现代社会的压力常常使人感到疲惫不堪，许多人开始寻求各种方式来缓解压力并提升生活质量。一些人选择加入健身房，通过锻炼身体释放体内的内啡肽，获得精神上的满足；另一些人则投身于艺术创作，绘画、写作或音乐成为他们表达内心世界的重要途径。还有人喜欢旅行，探索未知的地方，体验不同的文化和风景，以此开阔眼界和丰富人生阅历。无论采取哪种方式，最重要的是找到适合自己的方法，享受生活带来的每一份快乐。',
    '每逢佳节倍思亲，当春节的脚步逐渐临近，人们心中那份对家的思念愈发浓烈，仿佛能穿越时空的距离。家庭成员们从各地赶回来团聚，准备丰盛的年夜饭，分享过去一年的经历和故事。长辈们围坐在一起打牌聊天，年轻人则忙着用手机拍摄照片和视频，记录下珍贵的家庭时刻。夜晚降临，烟花在天空中绽放，照亮了每一个角落，为这个特别的节日增添了更多的欢乐气氛。'
]

test_en = [
    'On a bright and breezy morning, Xiao Ming decided to leave his small apartment nestled among towering skyscrapes in the bustling city center and head for the leafy park nearby. He was in search of a serene place amidst the rapid pace of urban life, yearning for an escape from the cacophony and hustle that defined his daily existence. The air in the park was refreshingly crisp and fragrant, accompanied by the cheerful chirping of birds and the innocent laughter of children, as if time itself had paused to allow one to savor the rare moment of relaxation and peace. As he strolled along, he observed an elderly man practicing Tai Chi, his movements graceful yet deliberate, each stroke imbued with strength and harmony, which reminded him of his childhood days spent learning martial arts from his grandfather.',
    "During festive seasons, especially as the Lunar New Year approaches, people's longing for home becomes increasingly intense, as if it can transcend the distance across time and space. Family members converge from far and wide to reunite, preparing a sumptuous feast together, sharing stories and experiences from the past year. Elders gather around card games and lively chats, while the younger generation busies itself capturing photos and videos with their smartphones, documenting precious family moments. As night falls, fireworks light up the sky, illuminating every corner, adding a festive touch to this special occasion.",
    "Modern society's pressures often leave individuals feeling exhausted, prompting many to seek various methods to alleviate stress and enhance quality of life. Some opt to join gyms, releasing endorphins through physical exercise and achieving mental satisfaction; others immerse themselves in artistic creation, finding painting, writing, or music as vital outlets for expressing their inner worlds. Travel enthusiasts explore uncharted territories, experiencing diverse cultures and landscapes, broadening their horizons and enriching their lives. Ultimately, the key is to find what suits oneself best, relishing the joy that life brings.",
    "The advancement of information technology has transformed how people communicate, allowing social media platforms to keep connections strong even over vast distances. Friends can instantly share updates and emotions via messaging apps; families can bridge gaps through video calls, making distant relatives feel close at hand. Yet, this convenience comes with challenges like privacy concerns and personal data security. Therefore, while enjoying the benefits of the web, we should also strengthen our awareness of self-protection and use online resources wisely.",
    "With the rapid evolution of technology, human lifestyles are undergoing profound changes, impacting our social interactions. Smartphones have enabled instant access to information and constant connectivity, but may also lead to fewer face-to-face exchanges. To adapt, we must strike a balance between online and offline social activities, leveraging digital tools without neglecting real-world relationships. Emerging technologies such as virtual reality (VR) and augmented reality (AR) promise to further reshape future social landscapes."
]

inputs = tokenizer(test_zh, return_tensors="pt", padding=True).to(model.device)
labels = inputs["input_ids"].clone()
labels[labels == tokenizer.pad_token_id] = -100
outputs = model(**inputs, labels=labels)
print('zh loss',outputs.loss.item())

inputs = tokenizer(test_en, return_tensors="pt", padding=True).to(model.device)
labels = inputs["input_ids"].clone()
labels[labels == tokenizer.pad_token_id] = -100
outputs = model(**inputs, labels=labels)
print('en loss',outputs.loss.item())

