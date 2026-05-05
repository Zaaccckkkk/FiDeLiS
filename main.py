import os
import argparse
import os
import json
import logging
import multiprocessing as mp
import datetime
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from src.evaluate_results import eval_result
from tqdm import tqdm
try:
   import wandb
except ImportError:
   wandb = None

try:
   from datasets import Dataset, load_dataset, load_from_disk
except ImportError:
   Dataset = None
   load_dataset = None
   load_from_disk = None
from src import utils
from src.llm_navigator import LLM_Navigator
from src.utils.data_types import Graph

try:
   import litellm
   litellm.set_verbose = False
except ImportError:
   litellm = None
set_verbose = False
now = datetime.datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")

def load_openai_key():
   if os.environ.get("OPENAI_API_KEY"):
      return
   config_path = "config.json"
   if os.path.exists(config_path):
      with open(config_path, "r") as f:
         config = json.load(f)
      api_key = config.get("OPENAI_API_KEY")
      if api_key and api_key != "your-api-key-here":
         os.environ["OPENAI_API_KEY"] = api_key


def disable_logging_during_run():
   logging.disable(logging.CRITICAL)
   

def prepare_dataset(sample):
   graph = utils.build_graph(sample["graph"])
   paths = utils.get_truth_paths(sample["q_entity"], sample["a_entity"], graph)
   if not paths or all(not path for path in paths): # if there is no path or all paths are empty
      sample["ground_paths"] = [["NA"]] # do not accept null sequence type, use "NA" instead
      sample["hop"] = 0
      return sample
   ground_paths = set()
   for path in paths:
      ground_paths.add(tuple([p[1] for p in path]))  # extract relation path
   sample["ground_paths"] = list(ground_paths) # [list(p) for p in ground_paths], [[], [], ...]
   sample["hop"] = len(list(ground_paths)[0])

   return sample


def prepare_crlt_dataset(sample):
   ground_paths = []
   for step in sample["reasoning_steps"]:
      fact = step.get("facts used in this step")
      if fact:
         ground_paths.append(fact)
      
   sample["ground_paths"] = ground_paths
   
   return sample


def parse_crlt_triples(triples_text):
   triples = []
   for match in re.finditer(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)", triples_text):
      h, r, t = match.groups()
      triples.append([h.strip(), r.strip(), t.strip()])
   return triples


def load_local_crlt_dataset(args):
   if Dataset is None:
      raise ImportError("The 'datasets' package is required. Install it with: pip install datasets")
   with open(args.crlt_path, "r", encoding="utf-8") as f:
      raw_dataset = json.load(f)

   processed = []
   for sample in raw_dataset:
      graph = parse_crlt_triples(sample.get("KG Triples", ""))
      q_entities = list(sample.get("KG Entities", {}).keys())
      answer = sample.get("answer")
      reasoning_steps = sample.get("Reasoning Steps", [])
      processed.append({
         "id": sample.get("id"),
         "question": sample.get("query", ""),
         "q_entity": q_entities,
         "a_entity": [str(answer)],
         "graph": graph,
         "hop": max(1, len(graph)),
         "reasoning_steps": reasoning_steps,
         "ground_paths": [step.get("facts used in this step") for step in reasoning_steps if step.get("facts used in this step")],
      })

   return Dataset.from_list(processed)


def data_processing(args):
   
   if args.d == "RoG-webqsp" or args.d == "RoG-cwq":
      if load_dataset is None:
         raise ImportError("The 'datasets' package is required. Install it with: pip install datasets")
      if os.path.isdir(args.data_path):
         input_file = os.path.join(args.data_path, args.d)
      else:
         input_file = f"{args.data_path.rstrip('/')}/{args.d}"
      output_file = os.path.join(args.save_cache, f"{args.d}_processed")
      dataset = load_dataset(input_file, split=args.split, cache_dir=args.save_cache)
      dataset = dataset.map(
         prepare_dataset,
         num_proc=args.N_CPUS,
      )
   
   elif args.d == "CR-LT-KGQA":   
      input_file = args.crlt_path if os.path.exists(args.crlt_path) else os.path.join(args.d)
      output_file = os.path.join(args.save_cache, f"{args.d}_processed")
      if os.path.exists(args.crlt_path):
         dataset = load_local_crlt_dataset(args)
         dataset = dataset.filter(
            lambda x: x.get("hop") > 0 and x.get("question") != "" and len(x.get("q_entity")) > 0 and len(x.get("a_entity")) > 0 and len(x.get("ground_paths")) > 0
         )
         if os.path.exists(output_file) == False:
            os.makedirs(output_file)
         dataset.save_to_disk(output_file)
         return dataset
      else:
         if load_dataset is None:
            raise ImportError("The 'datasets' package is required. Install it with: pip install datasets")
         dataset = load_dataset(input_file, split="train", cache_dir=args.save_cache)
      dataset = dataset.map(
         prepare_crlt_dataset,
         num_proc=args.N_CPUS,
      )

   dataset = dataset.filter(
         lambda x: x.get("hop") > 0 and x.get("question") != "" and len(x.get("q_entity")) > 0 and len(x.get("a_entity")) > 0 and len(x.get("ground_paths")) > 0, 
         num_proc=args.N_CPUS
      )
   dataset = dataset.filter(
         lambda x: x.get("q_entity") != None, 
         num_proc=args.N_CPUS
      )
   if os.path.exists(output_file) == False:
      os.makedirs(output_file)
      
   dataset.save_to_disk(output_file)
   return dataset


def init_embedding(data):
   id = data["id"]
   graph = Graph(
      args=args,
      graph=utils.build_graph(data["graph"]),
      cache_path=args.save_cache,
      id=id,
      embedding_method=args.embedding_model,
      replace=False
   )
   print(f"Embedding for {id} completed")


def write_selected_ids(dataset, args, output_dir):
   selected_ids_path = os.path.join(output_dir, "selected_ids.jsonl")
   with open(selected_ids_path, "w", encoding="utf-8") as f:
      for data in dataset:
         f.write(json.dumps({
            "dataset": args.d,
            "id": data.get("id"),
            "question": data.get("question", "")
         }) + "\n")


def select_sample(dataset, args, output_dir=None):
   if args.sample == -1:
      if output_dir is not None:
         write_selected_ids(dataset, args, output_dir)
      return dataset

   sample_size = min(args.sample, len(dataset))
   dataset = dataset.shuffle(seed=args.seed)
   dataset = dataset.select(range(sample_size))
   if output_dir is not None:
      write_selected_ids(dataset, args, output_dir)
   return dataset


def main(args):
   load_openai_key()
   if not os.environ.get("OPENAI_API_KEY"):
      raise ValueError("Set OPENAI_API_KEY in your environment or replace the placeholder in config.json before running inference.")
   if load_from_disk is None:
      raise ImportError("The 'datasets' package is required. Install it with: pip install datasets")

   output_dir = os.path.join(args.output_path, args.model_name, timestamp)
   
   if os.path.exists(output_dir) == False:
      os.makedirs(output_dir)
      
   logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(levelname)s - %(message)s',
      filename=os.path.join(output_dir,'detailed_process.log'),
      filemode='w',
   )
   if not args.debug:
      disable_logging_during_run() 
   
   wandb_run = None
   if args.use_wandb:
      if wandb is None:
         raise ImportError("wandb is not installed. Install it or omit --use_wandb.")
      wandb_run = wandb.init(project="rog", name=f"{args.d}-{args.model_name}-{args.sample}", config=args)
   
   # load the dataset
   cached_dataset_path = os.path.join(args.save_cache, f"{args.d}_processed")
   
   if os.path.exists(cached_dataset_path):
      dataset = load_from_disk(cached_dataset_path)
   else:
      print("Processing data...")
      dataset = data_processing(args)
      print("Data processing completed!")

   # generate embeddings
   if args.generate_embeddings:
      with ProcessPoolExecutor(max_workers=args.N_CPUS) as executor:
         # Using list to consume the results as they come in for tqdm to trackß
         executor.map(init_embedding, dataset)
      print("Embedding completed!")
      return
   
   # sample the dataset
   dataset = select_sample(dataset, args, output_dir)
   
   llm_navigator = LLM_Navigator(args)
   for data in tqdm(dataset, desc="Data Processing...", delay=0.5, ascii="░▒█"):
      
      if args.debug:         
         res, wandb_span = llm_navigator.beam_search(data) # run the beam search for each sample
         
      else:
         # run the beam search for each sample, catch the exception if it occurs
         try:
            res, wandb_span = llm_navigator.beam_search(data) # run the beam search for each sample
            
         except Exception as e:
            logging.error("Error occurred: {}".format(e))
            print("Error occurred: {}".format(e))
            json_str = json.dumps({"id": data['id'], "error": str(e)})
            with open(os.path.join(output_dir, "error_sample.jsonl"), "a") as f:
               f.write(json_str + "\n")
            continue
      
      if args.debug:
         for span in wandb_span:
            span.log(name="openai")
         
      json_str = json.dumps(res)
      with open(os.path.join(output_dir, f"{args.d}-{args.model_name}-{args.sample}.jsonl"), "a") as f:
         f.write(json_str + "\n")
      
   # evaluate
   llm_res, direct_ans_res = eval_result(os.path.join(output_dir, f"{args.d}-{args.model_name}-{args.sample}.jsonl"), cal_f1=True)
   
   if wandb_run:
      wandb.log({
         "llm_res": llm_res,
         "direct_ans_res": direct_ans_res
      })

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--N_CPUS", type=int, default=mp.cpu_count())
   parser.add_argument("--sample", type=int, default=-1)
   parser.add_argument("--seed", type=int, default=42)
   parser.add_argument("--data_path", type=str, default="rmanluo")
   parser.add_argument("--crlt_path", type=str, default=os.path.join("datasets", "crlt", "CR-LT-QA.json"))
   parser.add_argument("--d", "-d", type=str, choices=["RoG-webqsp", "RoG-cwq", "CR-LT-KGQA"], default="RoG-webqsp")
   parser.add_argument("--save_cache", type=str, default=os.path.join("datasets", "cache"))
   parser.add_argument("--split", type=str, default="test")
   parser.add_argument("--output_path", type=str, default="results")
   parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
   parser.add_argument("--top_n", type=int, default=30)
   parser.add_argument("--top_k", type=int, default=3)
   parser.add_argument("--max_length", type=int, default=3)
   parser.add_argument("--strategy", type=str, default="discrete_rating")
   parser.add_argument("--squeeze", type=bool, default=True)
   parser.add_argument("--verifier", type=str, default="deductive+planning")
   parser.add_argument("--disable_termination_verification", action=argparse.BooleanOptionalAction, default=True)
   parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
   parser.add_argument("--add_hop_information", action="store_true")
   parser.add_argument("--generate_embeddings", action="store_true")
   parser.add_argument("--alpha", type=float, default=0.3)
   parser.add_argument("--debug", action="store_true")
   parser.add_argument("--use_wandb", action="store_true")
   args = parser.parse_args()
   main(args)
