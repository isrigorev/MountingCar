"""
True Online SARSA(λ) с tile coding для Mountain Car.
Исправленная версия: корректная обработка truncated, чистая оценка.
"""

import warnings
warnings.filterwarnings("ignore", message=".*CRASHES ARE TO BE EXPECTED.*")

import numpy as np
import gymnasium as gym
from tiles3 import tiles, IHT



# ----------------------------- Параметры -----------------------------

NUM_TILINGS   = 8
IHT_SIZE      = 16384              # увеличил: 4096 мог переполняться
ALPHA         = 0.1 / NUM_TILINGS  # 0.0125 — безопаснее, чем 0.5/N
GAMMA         = 1.0
LAMBDA        = 0.9
EPSILON_START = 0.1
EPSILON_MIN   = 0.001
EPSILON_DECAY = 0.995

NUM_EPISODES  = 900
MAX_STEPS     = 1000

POS_MIN, POS_MAX = -1.2, 0.6
VEL_MIN, VEL_MAX = -0.07, 0.07
SCALE_POS = 10.0 / (POS_MAX - POS_MIN)
SCALE_VEL = 10.0 / (VEL_MAX - VEL_MIN)


# ----------------------------- Tile coding -----------------------------

iht = IHT(IHT_SIZE)
weights = np.random.uniform(0, 0.01, iht.size)
traces  = np.zeros(iht.size)


def mytiles(position, velocity, action):
    #делаем скейл состояния чтобы не получить понижение дискретизации за счет операции floor
    #возвращаем индексы активных тайлов для состояния (position, velocity) для заданного действия action
    return np.array(tiles(
        iht, NUM_TILINGS,
        [position * SCALE_POS, velocity * SCALE_VEL],
        [int(action)]
    ), dtype=np.int64)


def q_values(position, velocity):
    # возвращаем вектор Q(s, *) для каждого действия a
    return np.array([np.sum(weights[mytiles(position, velocity, a)])
                     for a in range(3)])


# ----------------------------- Политика -----------------------------

def epsilon_greedy_probs(position, velocity, epsilon):
    #переводим Q(s, a) в вероятностную e-жадную политику pi(a|s) - задаем плотность распределения выбора a.
    n = 3
    q = q_values(position, velocity) #вектор q для всех трех действий
    probs = np.full(n, epsilon / n) #инициализируем вероятности
    best_val = q.max() #находим действие с максимальной ценностью
    best_actions = np.flatnonzero(q == best_val) #индексы действий с максимальной ценностью - на случай нескольких
    probs[best_actions] += (1.0 - epsilon) / len(best_actions) #прибавляем жадную вероятность поровну ко всем лучшим действиям
    return probs


def sample_action(position, velocity, epsilon):
    #выбор действия по заданной плотности распределения
    return int(np.random.choice(3, p=epsilon_greedy_probs(position, velocity, epsilon)))


def greedy_action(position, velocity):
    return int(np.argmax(q_values(position, velocity)))


# ----------------------------- True Online SARSA(λ) -----------------------------

def true_online_sarsa_lambda(env, num_episodes=NUM_EPISODES):
    #функция реализует листинг алгоритма "Истинно онлайновый Sarsa(λ)" из главы 12.7 Барто-Саттона
    global weights, traces

    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        # уменьшаем вероятность исследования с каждым эпизодом
        epsilon = max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY ** ep))

        # сброс текущего состояния к начальному
        obs, _ = env.reset()
        pos, vel = float(obs[0]), float(obs[1])

        action = sample_action(pos, vel, epsilon) #выбор действия
        active = mytiles(pos, vel, action) #выбор индексов активных тайлов
        q_hat = np.sum(weights[active]) #Q(s, a) - текущая оценка ценности пары состояние-действие
        q_old = 0.0 # переменная алгоритма

        total_reward = 0.0 #переменная бенчмарка
        step = 0

        while True:
            # шаг среды реализует уравнения перехода между состояниями при заданном действии
            obs_next, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step += 1

            pos_next, vel_next = float(obs_next[0]), float(obs_next[1])

            if terminated:
                # терминальное состояние: Q(S',·) = 0
                # q_hat_next = 0 поэтому выражение для delta и weights упрощаются
                delta = reward - q_hat
                weights[active] += ALPHA * delta * traces[active]
                break

            if truncated:
                break

            action_next = sample_action(pos_next, vel_next, epsilon) #выбираем новое действие
            active_next = mytiles(pos_next, vel_next, action_next) #новые активные тайлы
            q_hat_next = np.sum(weights[active_next]) #оценка q(s',a')

            delta = reward + GAMMA * q_hat_next - q_hat #TD ошибка

            # True Online обновление
            # w<-w + alpha(delta + q - q_old) * z - alpha(q-q_old)x
            weights += ALPHA * (delta + q_hat - q_old) * traces #слагаемое для всех компонент вектора
            weights[active] -= ALPHA * (q_hat - q_old) #слагаемое для активных компонент

            # Замещающий след
            # traces *= GAMMA * LAMBDA
            # traces[active] = np.maximum(traces[active], 1.0)

            # Голландский след - показал значительно лучшие результаты чем замещающий след
            # z ← γλ·z + (1 − αγλ·zᵀφ)·φ
            z_dot_phi = np.sum(traces[active])       
            traces *= GAMMA * LAMBDA
            traces[active] += 1.0 - ALPHA * GAMMA * LAMBDA * z_dot_phi

            q_old = q_hat
            q_hat = q_hat_next
            active = active_next
            action = action_next
            pos, vel = pos_next, vel_next

            if step >= MAX_STEPS:
                break

        traces[:] = 0.0

        episode_returns.append(total_reward)
        episode_lengths.append(step)

        if (ep + 1) % 50 == 0:
            avg_ret = np.mean(episode_returns[-50:])
            avg_len = np.mean(episode_lengths[-50:])
            print(f"Эпизод {ep+1:4d} | ср. return {avg_ret:8.1f} | "
                  f"ср. длина {avg_len:6.1f} | ε={epsilon:.3f} | "
                  f"IHT={len(iht.dictionary)}/{IHT_SIZE}")

    return episode_returns, episode_lengths


# ----------------------------- Оценка -----------------------------

def evaluate(env, num_episodes=20):
    """Оценка жадной политики (ε=0). Не меняет веса."""
    successes = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        pos, vel = float(obs[0]), float(obs[1])
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < MAX_STEPS:
            a = greedy_action(pos, vel)
            obs, r, terminated, truncated, _ = env.step(a)
            pos, vel = float(obs[0]), float(obs[1])
            steps += 1
        if terminated:
            successes += 1
    return successes


def visualize(num_episodes=3, delay=0.02):
    """Смотреть, как жадная политика играет в окне.
    Создаёт СВОЙ env с render_mode='human' — не переиспользует обучающий."""
    import time
    env_vis = gym.make("MountainCar-v0", render_mode="human",
                       max_episode_steps=MAX_STEPS)
    for ep in range(num_episodes):
        obs, _ = env_vis.reset()
        pos, vel = float(obs[0]), float(obs[1])
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < MAX_STEPS:
            a = greedy_action(pos, vel)
            obs, r, terminated, truncated, _ = env_vis.step(a)
            pos, vel = float(obs[0]), float(obs[1])
            steps += 1
            time.sleep(delay)
        print(f"Эпизод {ep+1}: {'успех' if terminated else 'провал'}, шагов={steps}")
    env_vis.close()
# ----------------------------- Запуск -----------------------------

if __name__ == "__main__":
    np.random.seed(42)

    env = gym.make("MountainCar-v0", max_episode_steps=MAX_STEPS)

    returns, lengths = true_online_sarsa_lambda(env)

    print("\n--- Оценка обученной политики (ε=0) ---")
    succ = evaluate(env, 20)
    print(f"Успешных эпизодов: {succ}/20")
    print(f"IHT заполнен: {len(iht.dictionary)}/{IHT_SIZE}")

    env.close()

    visualize()