import argparse
from wandb_train import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="algebra2005")
    parser.add_argument("--model_name", type=str, default="akt")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    # parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--sequence_mode",
        type=str,
        choices=["original", "collapse"],
        default="original"
    )
    parser.add_argument("--max_concepts", type=int, default=8)
    parser.add_argument("--early_stop_rounds", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-3)

    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=0)
   
    args = parser.parse_args()

    params = vars(args)
    main(params)

# python wandb_akt_train.py --dataset_name=algebra2005 --model_name=akt --emb_type=qid --fold=0 --save_dir=saved_model --use_wandb=0 --add_uuid=0 --d_model=256 --d_ff=256 --n_blocks=1 --dropout=0.05 --learning_rate=1e-5

# python wandb_predict.py --save_dir=saved_model/algebra2005_qid_saved_model_3407_0_0.05_256_256_8_1_1e-05_64_None_8_0.1_0.1_0.0_True_0_0 --bz=256
