from IPython.display import clear_output
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder  # records videos of episodes
import numpy as np
import matplotlib.pyplot as plt  # Graphical library
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Configuring Pytorch
from collections import namedtuple, deque
from itertools import count
import random

clear_output()

random.seed(10)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, inputs, outputs, num_hidden, hidden_size):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(inputs, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) \
                                            for _ in range(num_hidden - 1)])
        self.output_layer = nn.Linear(hidden_size, outputs)

    def forward(self, x):
        x.to(device)

        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        return self.output_layer(x)


def optimize_model(memory,
                   BATCH_SIZE,
                   state_dim,
                   policy_net,
                   target_net,
                   GAMMA,
                   optimizer,
                   DDQL,
                   target_net_on):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # Alteration from original code to account for k frame stacking, now
    # treats a state as final if the last 4 elements are equal to zero.
    non_final_mask = torch.tensor(tuple(map(lambda s: torch.sum(s[0][len(s[0]) - 4:])
                                            .absolute().item() > 0, batch.next_state)),
                                  device=device,
                                  dtype=torch.bool)

    # Can safely omit the condition below to check that not all states in the
    # sampled batch are terminal whenever the batch size is reasonable and
    # there is virtually no chance that all states in the sampled batch are 
    # terminal. Similar change to account for k frame stacking final states
    if sum(non_final_mask) > 0:
        non_final_next_states = torch.cat([s for s in batch.next_state if torch.sum(
            s[0][len(s[0]) - 4:]).absolute().item() > 0])
    else:
        non_final_next_states = torch.empty(0, state_dim).to(device)

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    actions_test = policy_net(state_batch)
    # Compute V(s_{t+1}) for all next states.
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        # Once again can omit the condition if batch size is large enough
        if sum(non_final_mask) > 0:
            if not DDQL:
                # Single DQL case, with additional condition for using target
                # network updates (for final ablation question)
                if target_net_on:
                    next_state_values[non_final_mask] = target_net(
                        non_final_next_states).max(1)[0].detach()
                else:
                    next_state_values[non_final_mask] = policy_net(
                        non_final_next_states).max(1)[0].detach()
            else:
                # DDQL case, select action from policy network and then values from
                # target network
                policy_actions = \
                    policy_net(non_final_next_states).max(1)[1].view(-1, 1)
                next_state_values[non_final_mask] = \
                    target_net(non_final_next_states).gather(1, policy_actions).view(-1, )
        else:
            next_state_values = torch.zeros_like(next_state_values)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss using mean squared error loss criterion CHANGE IN REPORT
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # Limit magnitude of gradient for update step (THIS LIMITATION HAS BEEN REMOVED,
    # SLOWING DOWN RUN TIME BUT IMPROVING RESULTS)
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)

    optimizer.step()


def plot_total_rewards(N_episodes, total_rewards):
    plt.figure(2)
    episodes = np.arange(N_episodes)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Total Steps')
    plt.plot(episodes, np.array(total_rewards))
    print(f"Average Reward: {sum(total_rewards) / N_episodes}")


def select_action(state=None, current_eps=0, n_actions=2, policy_net=None):
    sample = random.random()
    eps_threshold = current_eps
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]],
                            device=device,
                            dtype=torch.long)


def eps_decay(NUM_EPISODES, EPS_START, EPS_END, decay_type):
    '''
    Function to produce different epsilon decay schedules
    '''

    eps_linear = np.linspace(EPS_START, EPS_END, NUM_EPISODES)
    eps_const = 0.6 * np.ones(eps_linear.shape)
    eps_glie = np.ones(eps_linear.shape)
    eps_factor = np.ones(eps_linear.shape)
    factor = np.power(EPS_END, EPS_START / NUM_EPISODES)

    for i in range(1, len(eps_linear)):
        eps_glie[i] = eps_glie[i - 1] / i
        eps_factor[i] = eps_factor[i - 1] * factor

    if decay_type == 'linear':
        return eps_linear
    elif decay_type == '1/k':
        return eps_glie
    elif decay_type == 'const':
        return eps_const
    else:
        return eps_factor


def train(NUM_EPISODES=100,
          BATCH_SIZE=128,
          GAMMA=0.99,
          EPS_START=0.9,
          EPS_END=0.05,
          EPS_DECAY='power',
          LR=0.0001,
          num_hidden_layers=2,
          size_hidden_layers=128,
          network_sync_freq=10,
          k=1,
          replays=10000,
          target_net_on=True,
          DDQL=False,
          record=False):
    '''
    Main training loop for agent,
    input: all hyperparameters and ablation/augmentation toggles
    return: (np array) total rewards for each episode
    '''

    # Get number of states and actions from gym action space
    env = gym.make("CartPole-v1")
    env.reset()
    state_dim = k * len(env.state)  # x, x_dot, theta, theta_dot
    n_actions = env.action_space.n
    env.close()

    # Initialise two networks, policy net and identical target net
    policy_net = DQN(state_dim,
                     n_actions,
                     num_hidden_layers,
                     size_hidden_layers).to(device)
    target_net = DQN(state_dim,
                     n_actions,
                     num_hidden_layers,
                     size_hidden_layers).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), LR)
    memory = ReplayBuffer(replays)

    # Empty list to append total rewards for each episode to
    # (same as duration of episode)
    durations = []
    epsilon = EPS_START

    # Use custom function to get decay schedule
    eps_schedule = eps_decay(NUM_EPISODES, EPS_START, EPS_END, EPS_DECAY)

    for i_episode in range(NUM_EPISODES):

        if i_episode % 20 == 0:
            print("episode ", i_episode, "/", NUM_EPISODES)

        # Initialize the environment and state
        env.reset()
        state = torch.tensor(env.state).float().unsqueeze(0).to(device)

        # Initialise frame stacker and set first k frames as initial state
        k_states = deque([], maxlen=k)
        for i in range(k):
            k_states.append(state)

        for t in count():
            flattened_k_states = torch.stack(
                list(k_states)).reshape(-1).unsqueeze(0)  # Flatten
            action = select_action(flattened_k_states,
                                   epsilon,
                                   n_actions,
                                   policy_net)  # Select action from network
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            if not done:
                next_state = torch.tensor(
                    env.state).float().unsqueeze(0).to(device)
            else:
                # If terminal set next state as zeros
                next_state = torch.zeros_like(state)

            # Store the transition in memory
            # Append state to frame stacker deque, pushing out oldest frame
            k_states.append(next_state)
            next_flattened_k_states = torch.stack(
                list(k_states)).reshape(-1).unsqueeze(0)  # Flatten k states again
            memory.push(flattened_k_states,
                        action,
                        next_flattened_k_states,
                        reward)  # Push (s,a,s',r) to memory

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory,
                           BATCH_SIZE,
                           state_dim,
                           policy_net,
                           target_net,
                           GAMMA,
                           optimizer,
                           DDQL,
                           target_net_on)
            if done:
                break

        # Move onto next epsilon value in schedule
        epsilon = eps_schedule[i_episode]

        durations.append(t)

        # Sync target network with policy net every set number of episodes
        if i_episode % network_sync_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')

    env.close()

    # Plot rewards against episodes
    # plot_total_rewards(NUM_EPISODES, durations)
    # plt.show()

    # If record toggle on, record the episode
    if record:
        record_cart(policy_net, k)

    return durations


def record_cart(policy_net, k):
    '''
    Record function separated to be toggled in main training loop, if you want
    to see the video.
    '''
    env = gym.make("CartPole-v1")
    file_path = 'video/video.mp4'
    recorder = VideoRecorder(env, file_path)

    observation = env.reset()
    done = False

    state = torch.tensor(env.state).float().unsqueeze(0)
    k_states = deque([], maxlen=k)
    for i in range(k):
        k_states.append(state)

    duration = 0

    while not done:
        recorder.capture_frame()

        # Select and perform an action
        flattened_k_states = torch.stack(
            list(k_states)).reshape(-1).unsqueeze(0)
        action = select_action(flattened_k_states,
                               current_eps=0,
                               n_actions=2,
                               policy_net=policy_net)

        observation, reward, done, _ = env.step(action.item())
        duration += 1
        reward = torch.tensor([reward], device=device)

        # Observe new state
        state = torch.tensor(env.state).float().unsqueeze(0)
        k_states.append(state)

    recorder.close()
    env.close()
    print("Episode duration: ", duration)


def plot_learning_rate_full(total_rewards):
    '''
    Function to plot learning curves for a number of differnet training runs,
    taking mean and standard deviation and plotting this on the figure as a
    line with errorbars.
    Total rewards given as a numpy array (R x N) with each row as a full
    raining run under certain conditions:
    R = number of runs
    N = number of episodes per run
    '''
    n_episodes = np.shape(total_rewards)[1]
    episodes = np.arange(0, n_episodes, 1)

    # Take mean and std across diffent runs
    mean = np.mean(total_rewards, axis=0)
    std = np.std(total_rewards, axis=0)

    # MEAN
    x = episodes
    y1 = mean
    y2 = std

    # calculate polynomial
    z1 = np.polyfit(x, y1, 30)
    z2 = np.polyfit(x, y2, 30)
    f1 = np.poly1d(z1)
    f2 = np.poly1d(z2)

    # calculate new x's and y's
    x_new = np.linspace(x[0], x[-1], 500)
    y_new1 = f1(x_new)
    y_new2 = f2(x_new)

    plt.plot(x, mean, '-', markersize=1)
    plt.errorbar(x_new,
                 y_new1,
                 yerr=y_new2,
                 fmt='none',
                 color='blue',
                 ecolor='lightblue',
                 elinewidth=3,
                 capsize=0,
                 label="mean/std")

    plt.xlim([x[0] - 1, x[-1] + 1])
    plt.xlabel('Episode')
    plt.ylabel("Total Rewards")
    plt.legend()
    plt.title("Learning Curve")
    plt.show()


def plot_k_learning_rates(k_total_rewards):
    '''
    Function to plot mean learning curves for different values of k.
    k_total_rewards given as a numpy array (R x N) with each row as a full
    training run under certain conditions:
    R = number of runs
    N = number of episodes per run
    Also used for ablation and augmentation experiment
    '''

    # Select legend labels
    k_values = ["k=1", "k=2", "k=3", "k=4"]
    alterations = ["Normal", "No target network", "No replay buffer", "DDQL"]

    for i, total_rewards in enumerate(k_total_rewards):
        n_episodes = np.shape(total_rewards)[1]
        episodes = np.arange(0, n_episodes, 1)

        mean = np.mean(total_rewards, axis=0)

        x = episodes
        y = mean

        plt.plot(x, y, label=f"{alterations[i]}")

    plt.xlabel('Episode')
    plt.ylabel("Total Rewards")
    plt.legend()
    plt.title("Learning Curve")
    plt.show()


def plot_replay_deviation(replays_experiment, total_rewards):
    '''
    Function to plot standard deviation against size of replay buffer using a
    log x axis.
    total_rewards: given as a numpy array (R x N) with each row as a full
    training run under certain conditions
    R = number of runs
    N = number of episodes per run

    replays experiment: given as list of replay buffer sizes
    '''

    n_episodes = np.shape(total_rewards)[1]
    n_runs = np.shape(total_rewards)[0]
    max_reward = np.max(total_rewards)
    level = 0.67 * max_reward
    stds = []

    for run in range(n_runs):
        rewards = total_rewards[run, :]
        std = np.sqrt(np.sum((rewards - level) ** 2) / n_episodes)
        stds.append(std)

    x = replays_experiment
    y = np.array(stds)

    plt.plot(x, y)

    plt.xlabel('Replay Buffer Size')
    plt.xscale('log')
    plt.ylabel("Standard Deviation (From 67% Max Reward)")
    plt.legend()
    plt.title("Replay Buffer Deviations")
    plt.show()


# LEARNING CURVE
def run_learning_curve(NUM_EPISODES, N_RUNS, k, replays, target, DDQL):
    '''
    Hyperparameters selected here. Train agent over specified number of N_RUNS
    and plot the averaged learning curve return total_rewards as an an (R x N)
    matrix:
    R = number of runs
    N = number of episodes per run
    '''
    total_rewards = np.zeros(NUM_EPISODES)

    for run in range(N_RUNS):
        print(f"Training run: {run}")
        rewards = train(NUM_EPISODES,
                        BATCH_SIZE=32,
                        GAMMA=0.99,
                        EPS_START=0.99,
                        EPS_END=0.05,
                        EPS_DECAY='power',
                        LR=0.0001,
                        num_hidden_layers=2,
                        size_hidden_layers=128,
                        network_sync_freq=10,
                        k=k,
                        replays=replays,
                        target_net_on=target,
                        DDQL=DDQL,
                        record=True)

        total_rewards = np.vstack((total_rewards, np.array(rewards)))

    total_rewards = np.delete(total_rewards, (0), axis=0)
    plot_learning_rate_full(total_rewards)

    return total_rewards


# REPLAYS EXPERIMENT
def run_replays_experiment(NUM_EPISODES, replays_experiment):
    '''
    Train agent over different replay buffer sizes as a list
    (replay_experiment), and plot results. Hyperparameters selected here.
    '''
    total_rewards = np.zeros(NUM_EPISODES)

    for run in range(len(replays_experiment)):
        replays = replays_experiment[run]
        print(f"Replay Buffer: {replays}")
        rewards = train(NUM_EPISODES=200,
                        BATCH_SIZE=32,
                        GAMMA=0.99,
                        EPS_START=0.99,
                        EPS_END=0.05,
                        EPS_DECAY='power',
                        LR=0.0001,
                        num_hidden_layers=2,
                        size_hidden_layers=128,
                        network_sync_freq=10,
                        k=3,
                        replays=replays,
                        DDQL=False,
                        record=False)

        total_rewards = np.vstack((total_rewards, np.array(rewards)))

    total_rewards = np.delete(total_rewards, (0), axis=0)
    plot_replay_deviation(replays_experiment, total_rewards)


# K EXPERIMENT
def run_k_experiment(NUM_EPISODES, N_RUNS, k_experiment):
    '''
    Train agent over different k values (number of frames to stack) as a list
    (k_experiment), run each for N_RUNS and plot mean results.
    '''
    k_total_rewards = []
    total_rewards = np.zeros(NUM_EPISODES)

    for run in range(len(k_experiment)):
        k = k_experiment[run]
        print(f"k value: {k}")

        total_rewards = run_learning_curve(NUM_EPISODES,
                                           N_RUNS,
                                           k,
                                           replays=10000,
                                           target=True,
                                           DDQL=True)
        k_total_rewards.append(total_rewards)

    plot_k_learning_rates(k_total_rewards)


# ABLATION/AUGMENTATION EXPERIMENT
def run_ab_experiment(NUM_EPISODES, N_RUNS):
    '''
    Train agent ablating and augmenting different features. Run each for N_RUNS
    and plot all on same graph.
    '''
    altered_total_rewards = []
    total_rewards = np.zeros(NUM_EPISODES)

    for run in range(4):
        if run == 0:
            print("Normal run")
            total_rewards = run_learning_curve(NUM_EPISODES,
                                               N_RUNS,
                                               k=1,
                                               replays=10000,
                                               target=True,
                                               DDQL=False)
        elif run == 1:
            print("Ablating target network feature")
            total_rewards = run_learning_curve(NUM_EPISODES,
                                               N_RUNS,
                                               k=1,
                                               replays=10000,
                                               target=False,
                                               DDQL=False)
        elif run == 2:
            print("Ablating replay buffer")
            total_rewards = run_learning_curve(NUM_EPISODES,
                                               N_RUNS,
                                               k=1,
                                               replays=1,
                                               target=True,
                                               DDQL=False)
        elif run == 3:
            print("Implementing DDQN")
            total_rewards = run_learning_curve(NUM_EPISODES,
                                               N_RUNS,
                                               k=1,
                                               replays=10000,
                                               target=True,
                                               DDQL=True)

        altered_total_rewards.append(total_rewards)

    plot_k_learning_rates(altered_total_rewards)


# Run various 'experiments' in main run code below. Uncomment experiments that
# are not needed.

# Q1 IMPLEMENT DQN SOLUTION
run_learning_curve(NUM_EPISODES=200,
                   N_RUNS=10,
                   k=1,
                   replays=10000,
                   target=True,
                   DDQL=False)

# Q2 Hyperparameters of the DQN
# Run replays experiment for chosen parameters
# run_replays_experiment(NUM_EPISODES=200,
#                        replays_experiment = [1, 10, 100, 1000, 10000, 100000])

# Run k experiment for chosen parameters
# run_k_experiment(NUM_EPISODES=300, N_RUNS=1, k_experiment=[1, 2, 3, 4])

# Q3 Ablation/Augmentation Experiements
# Run ablation/augmentation experiment for chosen parameters
# run_ab_experiment(NUM_EPISODES=200, N_RUNS=3)
