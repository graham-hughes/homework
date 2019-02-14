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

def load_data(expert_data):
    with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
        data = pickle.load(expert_data, f, pickle.HIGHEST_PROTOCOL)
        return data


# Given expert expert data and args flags, returns a trained model
def train_model(expert_data, args):
    observations = expert_data['observations']

    actions = expert_data['actions']
    actions_size = len(actions[0])


    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    model = tf.keras.Sequential([
      tf.keras.layers.Dense(observation_size, activation=tf.nn.relu, input_shape=(observations.shape[1],)),  # input shape required
      tf.keras.layers.Dense(10, activation=tf.nn.relu),
      tf.keras.layers.Dense(actions_size)
    ])

    model.compile(loss='mean_squared_error', optimizer=tf.optimizers.Adam(lr=args.learning_rate))

    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        args.samples_per_epoch,
                        args.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)
    return model

def predict(model, data):
    return model.predict(
        data,
        batch_size=None,
        verbose=0,
        steps=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    )



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

    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading expert data')

    expert_data = load_data(args.envname)

    model = train_model(expert_data, args)

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns_expert = []
        observations_expert = []
        actions_expert = []

        returns_cloning = []
        observations_cloning = []
        actions_cloning = []
        for i in range(args.num_rollouts):
            print('iter', i)
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
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns_expert.append(totalr)

        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = model(obs[None,:])
                observations_cloning.append(obs)
                actions_cloning.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break

            returns_cloning.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()