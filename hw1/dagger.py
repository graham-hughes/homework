#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python behavior_cloning.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

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
def load_data(args):
    with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data


# Given expert expert data and args flags, returns a trained model
def create_model(args):
    expert_data = load_data(args)

    observations = expert_data['observations']

    actions = expert_data['actions']
    actions_size = len(actions[0])
    
    print(observations.shape)
    print(actions.shape)

    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(observations, actions, test_size=args.test_size, random_state=0)

    model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(observations.shape[1],)),  # input shape required
      tf.keras.layers.Dense(10, activation=tf.nn.relu),
      tf.keras.layers.Dense(actions.shape[2], activation='linear')
    ])

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=args.learning_rate))

    return model

def fit_model(args, observations, actions, model, X_train, y_train, X_valid, y_valid):
    train_input = X_train.reshape(X_train.shape[0], observations.shape[1])
    test_input = X_valid.reshape(X_valid.shape[0], observations.shape[1])
    train_output = y_train.reshape(y_train.shape[0], actions.shape[2])
    test_output = y_valid.reshape(y_valid.shape[0], actions.shape[2])

    model.fit(x=train_input, y=train_output, validation_data=(test_input, test_output), verbose=1, batch_size=args.batch_size, nb_epoch=args.nb_epoch)

    return model


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
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    expert_data = load_data(args)
    observations = expert_data['observations']
    actions = expert_data['actions']

    mean_returns = []
    standard_deviations = []

    model = create_model(args)

    # DAgger loop
    for i in range(5):

        # Split data
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(observations, actions, test_size=args.test_size, random_state=0)

        model = fit_model(args, observations, actions, model, X_train, y_train, X_valid, y_valid)
    
        with tf.Session():
            tf_util.initialize()

            import gym
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            returns = []
            new_observations = []
            new_actions = []

            for i in range(args.num_rollouts):
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    obs = np.array(obs)
                    expert_action = policy_fn(obs[None,:])
                    obs = np.expand_dims(obs, 0)
                    action = (model.predict(obs, batch_size=64, verbose=0))

                    new_observations.append(obs)
                    new_actions.append(action)
                    obs, r, done, _ = env.step(action)

                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps >= max_steps:
                        break
                mean_returns.append(totalr)

            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))

            mean_returns.append(np.mean(returns))
            standard_deviations.append(np.std(returns))

        new_observations = np.array(new_observations)
        new_actions = np.array(new_actions)

        new_observations = new_observations.reshape((new_observations.shape[0], obs_data.shape[1]))
        observations = np.concatenate((observations, new_observations))
        actions = np.concatenate((actions, new_actions))

    print(mean_returns)
    print(standard_deviations)


if __name__ == '__main__':
    main()
