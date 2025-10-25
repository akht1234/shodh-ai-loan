import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset
import tensorflow as tf

def make_rewards(loan_amnt_arr, int_rate_arr, outcomes, actions):
    int_frac = np.array(int_rate_arr, dtype=np.float32) / 100.0
    loan_amnt = np.array(loan_amnt_arr, dtype=np.float32)
    outcomes = np.array(outcomes, dtype=np.int32)
    actions = np.array(actions, dtype=np.int32)
    rewards = np.zeros_like(actions, dtype=np.float32)
    approve_and_paid = (actions == 1) & (outcomes == 0)
    rewards[approve_and_paid] = loan_amnt[approve_and_paid] * int_frac[approve_and_paid]
    approve_and_default = (actions == 1) & (outcomes == 1)
    rewards[approve_and_default] = - loan_amnt[approve_and_default]
    return rewards

processed_df = pd.read_csv("/kaggle/working/lc_processed_sample.csv")

try:
    X = processed_df.drop(columns=['target']).values.astype('float32')
except:
    num_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != 'target']
    X = processed_df[num_cols].fillna(0).values.astype('float32')

y = processed_df['target'].values.astype(int)
loan_amnt = processed_df['loan_amnt'].values
int_rate = processed_df['int_rate'].values

X_train, X_test, y_train, y_test, loan_train, loan_test, int_train, int_test = train_test_split(
    X, y, loan_amnt, int_rate, test_size=0.2, random_state=42, stratify=y
)

actions_logged_train = np.ones((X_train.shape[0],), dtype=int)
rewards_logged_train = make_rewards(loan_train, int_train, y_train, actions_logged_train)
terminals_train = np.ones_like(rewards_logged_train, dtype=bool)

obs_train_aug = np.concatenate([X_train, X_train], axis=0)
actions_train_aug = np.concatenate([actions_logged_train, np.zeros_like(actions_logged_train)], axis=0)
rewards_train_aug = np.concatenate([rewards_logged_train, np.zeros_like(rewards_logged_train)], axis=0)
terminals_train_aug = np.concatenate([terminals_train, np.ones_like(terminals_train)], axis=0)

dataset = MDPDataset(
    observations=obs_train_aug,
    actions=actions_train_aug,
    rewards=rewards_train_aug,
    terminals=terminals_train_aug
)

config = DiscreteCQLConfig(
    learning_rate=1e-4,
    batch_size=256
)

cql = config.create(device="cuda:0")

n_epochs = 50
n_steps = n_epochs * (dataset.transition_count // 256)

cql.fit(
    dataset,
    n_steps=n_steps,
    n_steps_per_epoch=dataset.transition_count // 256,
    show_progress=True
)

policy_actions = cql.predict(X_test)
policy_rewards = make_rewards(loan_test, int_test, y_test, policy_actions)
avg_return_per_loan = policy_rewards.mean()
total_expected_return = policy_rewards.sum()
n_test = len(policy_rewards)

print(f"Test set size: {n_test}")
print(f"Avg reward per loan under learned policy: {avg_return_per_loan:.4f}")
print(f"Total expected return on test set (sum): {total_expected_return:.2f}")

always_rewards = make_rewards(loan_test, int_test, y_test, np.ones_like(policy_actions))
print("Always-approve avg reward:", always_rewards.mean(), "sum:", always_rewards.sum())

deny_rewards = make_rewards(loan_test, int_test, y_test, np.zeros_like(policy_actions))
print("Always-deny avg reward:", deny_rewards.mean(), "sum:", deny_rewards.sum())

try:
    from tensorflow.keras.models import load_model
    model = load_model("/kaggle/working/dl_model.h5")
    probs = model.predict(X_test)
    th = 0.5
    dl_actions = (probs.reshape(-1) < th).astype(int)
    dl_rewards = make_rewards(loan_test, int_test, y_test, dl_actions)
    print("DL-threshold avg reward:", dl_rewards.mean(), "sum:", dl_rewards.sum())
except:
    print("DL model not found — skipping DL baseline.")

