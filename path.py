import os
import torch

from torch_geometric.data import Data
from ultra.tasks import build_relation_graph
from datasets import load_dataset
from tqdm import tqdm


from ultra.models import Ultra
import yaml
import easydict
import json

def load_yaml(args):
    with open(args.file_path, 'r') as file:
        content = file.read()
        content = content.replace("{{ dataset }}", "\""+args.dataset+"\"")
        content = content.replace("{{ gpus }}", "\""+str(args.gpus)+"\"")
        content = content.replace("{{ ckpt }}", "\""+args.checkpoint+"\"")
        content = content.replace("{{ bpe }}", "\""+args.bpe+"\"")
        content = content.replace("{{ epochs }}", "\""+args.epochs+"\"")
        config = yaml.safe_load(content)
        config = easydict.EasyDict(config)
    return config


def extract_relation_path(paths, relation_vocab, relation_paths=set()):
    num_relation = len(relation_vocab)
    for path in paths:
        relation_list = []
        for h, t, r in path:
            r_name = relation_vocab[r % num_relation]
            # if r >= num_relation:
            #     r_name += "^(-1)"
            relation_list.append(r_name)
        relation_paths.add(tuple(relation_list))  # extract relation path
    return relation_paths


def build_vocab(graph):
    triplets = []
    inv_entity_vocab, inv_rel_vocab = {}, {}
    entity_cnt, rel_cnt = len(inv_entity_vocab), len(inv_rel_vocab)
    
    for triple in graph:
        u, r, v = triple
        if u not in inv_entity_vocab:
            inv_entity_vocab[u] = entity_cnt
            entity_cnt += 1
        if v not in inv_entity_vocab:
            inv_entity_vocab[v] = entity_cnt
            entity_cnt += 1
        if r not in inv_rel_vocab:
            inv_rel_vocab[r] = rel_cnt
            rel_cnt += 1
        u, r, v = inv_entity_vocab[u], inv_rel_vocab[r], inv_entity_vocab[v]

        triplets.append((u, v, r))
    return triplets, inv_entity_vocab, inv_rel_vocab


if __name__ == "__main__":
    class args:
        data_name = "cwq"
        seed=1024
        dataset = 'CustomTransductiveDataset'
        gpus =0
        checkpoint = '/home/bonbak/ULTRA/ckpts/ultra_50g.pth'
        file_path = '/home/bonbak/ULTRA/config/transductive/inference.yaml'
        bpe = 'null'
        epochs = '0'
        rel_topk = 1
        path_topk = 2
        max_hop = 2 if data_name == "webqsp" else 4
        save_dir = f"/home/bonbak/ULTRA/paths/{data_name}/rel{rel_topk}-path{path_topk}"
        

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    question_dataset = load_dataset(f"rmanluo/RoG-{args.data_name}")['train']
    cfg = load_yaml(args)
    model = Ultra(
        rel_model_cfg=cfg.model.relation_model,
        entity_model_cfg=cfg.model.entity_model,
    )

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    device = f"cuda:"+str(args.gpus)
    model = model.to(device)

    model.eval()

    with open(f'{args.save_dir}/{args.data_name}_train.jsonl', 'w') as f: 
        f.write('')
    with open(f'{args.save_dir}/{args.data_name}_qid2path.jsonl', 'w') as f:
        f.write('')
        
    for row in tqdm(question_dataset):
        if len(row['q_entity']) != 1 or len(row['a_entity']) != 1:
            continue
        
        triplets, inv_entity_vocab, inv_rel_vocab = build_vocab(row['graph'])
        if len(triplets) == 0: continue
        relation_vocab = {id:rel for rel,id in inv_rel_vocab.items()}

        # build graph
        num_node = len(inv_entity_vocab)
        num_relations = len(inv_rel_vocab)

        
        edges = torch.tensor([[t[0], t[1]] for t in triplets], dtype=torch.long).t()
        etypes = torch.tensor([t[2] for t in triplets])

        edges = torch.cat([edges, edges.flip(0)], dim=1)
        etypes = torch.cat([etypes, etypes+num_relations])

        data = Data(edge_index=edges, edge_type=etypes, num_nodes=num_node, num_relations=num_relations*2)
        data = build_relation_graph(data)

        relation_paths = set()

        # find path
        for q in row['q_entity']:
            if q not in inv_entity_vocab:
                continue
            for a in row['a_entity']:
                if a not in inv_entity_vocab:
                    continue

                infer_edges = torch.tensor([[inv_entity_vocab[q], inv_entity_vocab[a]] for _ in range(num_relations)], dtype=torch.long).t()
                infer_etypes = torch.arange(num_relations)

                data.target_edge_index = infer_edges
                data.target_edge_type = infer_etypes

                # find strongest link
                with torch.no_grad():
                    data = data.to(device)
                    batch = torch.cat([data.target_edge_index, data.target_edge_type.unsqueeze(0)]).t()

                    triples = batch.unsqueeze(1)
                    pred = model(data, triples)
                    values, indices = pred.squeeze().topk(k=args.rel_topk)

                    inv_pred = model(data, triples[:, :, [1, 0, 2]])
                    inv_values, inv_indices = inv_pred.squeeze().topk(k=args.rel_topk)

                for i in range(args.rel_topk):
                    paths, weights = model.visualize(data, batch[indices[i]].unsqueeze(0).unsqueeze(0))
                    relation_paths = extract_relation_path(paths[:args.path_topk], relation_vocab, relation_paths)
                    
                    inv_paths, inv_weights = model.visualize(data, batch[inv_indices[i]].unsqueeze(0).unsqueeze(0))
                    relation_paths = extract_relation_path(inv_paths[:args.path_topk], relation_vocab, relation_paths)
                
                del batch, triples, pred, inv_pred, values, indices, inv_values, inv_indices
                torch.cuda.empty_cache()
        
        del data
        torch.cuda.empty_cache()
        
        relation_paths = [list(rel_path) for rel_path in relation_paths if len(rel_path) <= args.max_hop]
    
        qid2path = {}
        qid2path['qid'] = row['id']
        qid2path['path'] = relation_paths
        with open(f'{args.save_dir}/{args.data_name}_qid2path.jsonl', 'a') as f:
            f.write(json.dumps(qid2path) + '\n')

        result_dict = {}
        result_dict['question'] = row['question']
        with open(f'{args.save_dir}/{args.data_name}_train.jsonl', 'a') as f:
            for rel_path in relation_paths:
                if len(rel_path) == 0: continue
                result_dict['path'] = rel_path
                f.write(json.dumps(result_dict) + '\n')