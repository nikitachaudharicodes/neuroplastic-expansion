import os
import json
import argparse
import torch
import gymnasium as gym
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Import actor class
from modules_TD3 import MLPActor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_actor(model_dir, state_dim, action_dim, max_action, hidden_dim=256):
    """
    Reconstructs and loads the trained actor.
    """
    actor_path = os.path.join(model_dir, "actor")
    actor_final_path = os.path.join(model_dir, "actor_final")

    if os.path.exists(actor_final_path):
        ckpt = torch.load(actor_final_path, map_location=DEVICE)
    elif os.path.exists(actor_path):
        ckpt = torch.load(actor_path, map_location=DEVICE)
    else:
        raise FileNotFoundError("No actor or actor_final checkpoint found.")

    actor = MLPActor(state_dim, action_dim, max_action, hidden_dim).to(DEVICE)
    actor.load_state_dict(ckpt)
    actor.eval()
    return actor


def evaluate(env, actor, episodes, record_video=False, video_dir=None):
    """
    Runs episodes and returns list of cumulative rewards.
    """
    rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done, truncated = False, False
        ep_reward = 0

        while not (done or truncated):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action = actor(state_t).cpu().numpy()[0]

            next_state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            state = next_state

        rewards.append(ep_reward)
        print(f"Episode {ep+1} reward: {ep_reward:.2f}")

    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--exp_id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=256)
    args = parser.parse_args()

    # -------------------------------------------------------
    # Locate training directory
    # -------------------------------------------------------
    model_dir = f"./results/{args.exp_id}_{args.env}/{args.seed}/model/"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading models from: {model_dir}")

    # -------------------------------------------------------
    # Init environment
    # -------------------------------------------------------
    if args.record_video:
        video_path = f"videos/{args.exp_id}/{args.seed}/"
        os.makedirs(video_path, exist_ok=True)

        env = gym.make(args.env, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_path)
    else:
        env = gym.make(args.env)

    env.reset(seed=args.seed)

    # Extract dims
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # -------------------------------------------------------
    # Load actor
    # -------------------------------------------------------
    actor = load_actor(
        model_dir,
        state_dim,
        action_dim,
        max_action,
        hidden_dim=args.hidden_dim
    )

    # -------------------------------------------------------
    # TensorBoard logging
    # -------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = f"runs/eval_{args.exp_id}_s{args.seed}_{timestamp}"
    writer = SummaryWriter(tb_dir)

    # -------------------------------------------------------
    # Evaluation loop
    # -------------------------------------------------------
    rewards = evaluate(env, actor, args.episodes)

    # Log to TensorBoard
    for i, r in enumerate(rewards):
        writer.add_scalar("Eval/EpisodeReward", r, i)

    # Save JSON
    out_json = {
        "exp_id": args.exp_id,
        "env": args.env,
        "seed": args.seed,
        "episodes": args.episodes,
        "rewards": rewards,
        "avg_reward": sum(rewards) / len(rewards)
    }

    os.makedirs(f"results_eval/{args.exp_id}/{args.seed}/", exist_ok=True)
    with open(f"results_eval/{args.exp_id}/{args.seed}/eval_results.json", "w") as f:
        json.dump(out_json, f, indent=4)

    print("\nEvaluation complete.")
    print(f"Average reward: {out_json['avg_reward']:.2f}")
    print(f"TensorBoard logs: {tb_dir}")
    if args.record_video:
        print(f"Videos saved to: {video_path}")


if __name__ == "__main__":
    main()
