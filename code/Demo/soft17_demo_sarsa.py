#!/usr/bin/env python3
"""
Soft17 - SARSA Demo
Abbatiello Simone
Nappi Vincenzo
Niemiec Francesco
"""

import random
from collections import defaultdict

# Percorsi
UPLOAD_PATH = "pics"

class BlackjackEnv:
    """Environment del Blackjack"""

    def __init__(self, num_decks=8):
        self.num_decks = num_decks
        self.reset_deck()

    def reset_deck(self):
        deck = []
        for _ in range(self.num_decks):
            for _ in range(4):
                deck.extend([11] + list(range(2, 11)) + [10, 10, 10])
        random.shuffle(deck)
        self.deck = deck

    def draw_card(self):
        if len(self.deck) < 20:
            self.reset_deck()
        return self.deck.pop()

    def get_hand_value(self, hand):
        value = sum(hand)
        aces = hand.count(11)
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1
        is_soft = (aces > 0 and value <= 21)
        return value, is_soft

    def is_bust(self, hand):
        value, _ = self.get_hand_value(hand)
        return value > 21

    def dealer_play(self, dealer_hand):
        while True:
            value, is_soft = self.get_hand_value(dealer_hand)
            if value > 21:
                break
            if value >= 17 and not is_soft:
                break
            if value == 17 and is_soft:
                dealer_hand.append(self.draw_card())
            elif value < 17:
                dealer_hand.append(self.draw_card())
            else:
                break
        return dealer_hand

    def reset(self):
        player_hand = [self.draw_card(), self.draw_card()]
        dealer_hand = [self.draw_card(), self.draw_card()]
        return {
            'player_hand': player_hand,
            'dealer_hand': dealer_hand,
            'dealer_showing': dealer_hand[0]
        }

    def step(self, state, action):
        player_hand = state['player_hand'].copy()
        dealer_hand = state['dealer_hand'].copy()
        dealer_showing = state['dealer_showing']
        done = False
        reward = 0
        info = {}

        if action == 1:  # HIT
            player_hand.append(self.draw_card())
            if self.is_bust(player_hand):
                reward = -1
                done = True
                info['outcome'] = 'player_bust'
            else:
                return {
                    'player_hand': player_hand,
                    'dealer_hand': dealer_hand,
                    'dealer_showing': dealer_showing
                }, reward, done, info

        elif action == 0:  # STAND
            done = True
            dealer_hand = self.dealer_play(dealer_hand)
            player_value, _ = self.get_hand_value(player_hand)
            dealer_value, _ = self.get_hand_value(dealer_hand)

            if self.is_bust(dealer_hand):
                reward = 1
                info['outcome'] = 'dealer_bust'
            elif player_value > dealer_value:
                reward = 1
                info['outcome'] = 'player_wins'
            elif player_value < dealer_value:
                reward = -1
                info['outcome'] = 'dealer_wins'
            else:
                reward = 0
                info['outcome'] = 'push'

        return {
            'player_hand': player_hand,
            'dealer_hand': dealer_hand,
            'dealer_showing': dealer_showing
        }, reward, done, info


def state_to_tuple(state, env):
    player_value, is_soft = env.get_hand_value(state['player_hand'])
    return (player_value, int(is_soft), state['dealer_showing'])


class SARSAAgent:
    def __init__(self, learning_rate=0.01, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.01):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: defaultdict(float))

    def get_best_action(self, state, env):
        state_key = state_to_tuple(state, env)
        if state_key not in self.q_table:
            return random.choice([0, 1])
        q_values = self.q_table[state_key]
        if not q_values:
            return random.choice([0, 1])
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def choose_action(self, state, env, training=False):
        player_value, _ = env.get_hand_value(state['player_hand'])
        if player_value >= 21:
            return 0
        if training and random.random() < self.epsilon:
            return random.choice([0, 1])
        else:
            return self.get_best_action(state, env)

    def get_q_values(self, state, env):
        state_key = state_to_tuple(state, env)
        if state_key in self.q_table:
            return self.q_table[state_key]
        return {0: 0.0, 1: 0.0}

    def get_reasoning(self, state, env):
        player_value, is_soft = env.get_hand_value(state['player_hand'])
        dealer_showing = state['dealer_showing']

        reasoning = []
        reasoning.append("=" * 50)
        reasoning.append("ANALISI SITUAZIONE")
        reasoning.append("=" * 50)
        reasoning.append(f"Mano giocatore: {state['player_hand']}")
        reasoning.append(f"Valore: {player_value} ({'soft' if is_soft else 'hard'})")
        reasoning.append(f"Carta visibile dealer: {dealer_showing}")
        reasoning.append("")

        if player_value >= 21:
            reasoning.append("DECISIONE: STAND (valore >= 21)")
            return "\n".join(reasoning)

        q_values = self.get_q_values(state, env)
        action = self.get_best_action(state, env)

        reasoning.append("Q-VALUES")
        reasoning.append(f"Q(STAND) = {q_values.get(0, 0.0):.4f}")
        reasoning.append(f"Q(HIT)   = {q_values.get(1, 0.0):.4f}")
        reasoning.append("")

        reasoning.append("DECISIONE AI")
        if action == 0:
            reasoning.append("STAND - Il modello preferisce fermarsi")
            reasoning.append(f"  Probabilmente il valore {player_value} è sufficiente")
        else:
            reasoning.append("HIT - Il modello consiglia di pescare")
            reasoning.append(f"  Il valore {player_value} è troppo basso")

        return "\n".join(reasoning)

    def train(self, env, num_episodes=500000, callback=None):
        for episode in range(num_episodes):
            state = env.reset()
            action = self.choose_action(state, env, training=True)
            done = False
            steps = 0

            while not done and steps < 50:
                next_state, reward, done, info = env.step(state, action)

                if done:
                    state_key = state_to_tuple(state, env)
                    current_q = self.q_table[state_key][action]
                    new_q = current_q + self.lr * (reward - current_q)
                    self.q_table[state_key][action] = new_q
                else:
                    next_action = self.choose_action(next_state, env, training=True)
                    state_key = state_to_tuple(state, env)
                    next_state_key = state_to_tuple(next_state, env)
                    current_q = self.q_table[state_key][action]
                    next_q = self.q_table[next_state_key][next_action]
                    new_q = current_q + self.lr * (reward + self.gamma * next_q - current_q)
                    self.q_table[state_key][action] = new_q
                    state = next_state
                    action = next_action

                steps += 1

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if callback and (episode + 1) % 10000 == 0:
                callback(episode + 1, num_episodes)
