#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python behavior_cloning.py experts/Humanoid-v2.pkl Humanoid-v2 --compare_rollouts True

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import sklearn
import sklearn.model_selection
import tensorflow.keras.optimizers
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# Given expert expert data and args flags, returns a trained model
def train_model(args, expert_data):

    observations = expert_data['observations']

    actions = expert_data['actions']

    # Splits observations/actions into train/test
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(observations, actions, test_size=args.test_size, random_state=0)

    train_input = X_train.reshape(X_train.shape[0], observations.shape[1])
    test_input = X_valid.reshape(X_valid.shape[0], observations.shape[1])
    train_output = y_train.reshape(y_train.shape[0], actions.shape[2])
    test_output = y_valid.reshape(y_valid.shape[0], actions.shape[2])

    model = tf.keras.Sequential([
      tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(observations.shape[1],)),  # input shape required
      tf.keras.layers.Dense(32, activation=tf.nn.relu),
      tf.keras.layers.Dense(actions.shape[2], activation='linear')
    ])

    # Compile model with mean squared log loss and Adam optimizer, using accuracy for loss metric
    model.compile(loss='msle', optimizer='adam', metrics=['accuracy'])

    # Fits model
    model.fit(x=train_input, y=train_output, validation_data=(test_input, test_output), verbose=0, batch_size=args.batch_size, nb_epoch=args.nb_epoch)

    return model

# Takes in necessary args/policy_fn/model, along with:
# number of rollouts and hyperparameter (defaults available), returns mean_expert, std_expert, mean_cloning, std_cloning
def compare_model_expert(args, policy_fn, num_rollouts, env, max_steps):

    returns_expert = []
    observations_expert = []
    actions_expert = []

    returns_cloning = []
    observations_cloning = []
    actions_cloning = []

    # Run the expert policy and extract observations/actions/returns
    for i in range(num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(obs[None,:])
            observations_expert.append(obs)
            actions_expert.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps >= max_steps:
                break
        returns_expert.append(totalr)

    expert_data = {'observations': np.array(observations_expert),
                   'actions': np.array(actions_expert)}

    # Train model using the observations/actions generated by the above policy function over num_rollouts
    model = train_model(args, expert_data)

    # Run the trained model and extract observations/actions/returns
    for i in range(num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            obs = np.expand_dims(obs, 0)
            action = (model.predict(obs, batch_size=64, verbose=0))
            observations_cloning.append(obs)
            actions_cloning.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps >= max_steps:
                break

        returns_cloning.append(totalr)

    mean_expert = np.mean(returns_expert)
    std_expert = np.std(returns_expert)
    mean_cloning = np.mean(returns_cloning)
    std_cloning = np.std(returns_cloning)

    return mean_expert, std_expert, mean_cloning, std_cloning

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str) # Expert policy
    parser.add_argument('envname', type=str) # Task
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)

    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument('--samples_per_epoch', type=int,   default=20000)
    parser.add_argument('--nb_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--test_size', type=float, default=0.2)

    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')

    parser.add_argument('--compare_rollouts', type=bool, default=False)
    parser.add_argument('--compare_hyperparameter', type=bool, default=False)

    args = parser.parse_args()

    # Preload expert policy function
    policy_fn = load_policy.load_policy(args.expert_policy_file)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        if (args.compare_rollouts):

            row_labels = []
            cell_text = []
            column_labels = ['mean_expert', 'std_expert', 'mean_cloning', 'std_cloning']

            # Store means/stds for expert/cloning over varying rollouts
            for num_rollouts in range (25, 150, 25):
                print('num_rollouts', num_rollouts)

                mean_expert, std_expert, mean_cloning, std_cloning = compare_model_expert(args, policy_fn, num_rollouts, env, max_steps)

                row_labels.append('%d rollouts' % num_rollouts)
                cell_text.append([str(mean_expert), str(std_expert), str(mean_cloning), str(std_cloning)])
            
            fig, ax = plt.subplots()

            fig.patch.set_visible(False)
            ax.axis('off')
            ax.axis('tight')
            table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=column_labels, loc='center')

            fig.tight_layout(pad=5)
            plt.savefig(fname=os.path.join('rollout_comparisons', args.envname + '.png'), dpi=300)



if __name__ == '__main__':
    main()
