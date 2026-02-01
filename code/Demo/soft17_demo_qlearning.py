#!/usr/bin/env python3
"""
Soft17 - Q-Learning Demo
Abbatiello Simone
Nappi Vincenzo
Niemiec Francesco
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
from PIL import Image, ImageTk
import random
import threading
from collections import defaultdict
import time

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


class QLearningAgent:
    """Q-Learning Agent - differenza principale: usa max(Q) invece di Q(s',a') nell'update"""
    
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
        """Training Q-Learning - usa max(Q(s',a)) invece di Q(s',a') come SARSA"""
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            steps = 0

            while not done and steps < 50:
                action = self.choose_action(state, env, training=True)
                next_state, reward, done, info = env.step(state, action)

                state_key = state_to_tuple(state, env)
                current_q = self.q_table[state_key][action]

                if done:
                    # Update terminale
                    new_q = current_q + self.lr * (reward - current_q)
                else:
                    # Q-Learning: usa max(Q(s',a)) - differenza chiave con SARSA
                    next_state_key = state_to_tuple(next_state, env)
                    next_q_values = self.q_table[next_state_key]
                    max_next_q = max(next_q_values.values()) if next_q_values else 0.0
                    new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)

                self.q_table[state_key][action] = new_q
                state = next_state
                steps += 1

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if callback and (episode + 1) % 10000 == 0:
                callback(episode + 1, num_episodes)


class BlackjackGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Q-Learning Blackjack - Soft17")
        self.root.geometry("1100x650")
        self.root.configure(bg='#1a472a')

        self.env = BlackjackEnv(num_decks=8)
        self.agent = QLearningAgent(epsilon=0.01)
        self.state = None
        self.game_active = False
        self.dealer_revealed = False
        self.show_lock = True

        self.card_images = {}
        self.bg_images = {}

        self.create_ui()
        self.start_training()

    def start_training(self):
        """Avvia il training in un thread separato"""
        self.log_to_console("=== BENVENUTO AL BLACKJACK Q-LEARNING ===\n")
        self.log_to_console("Inizializzazione in corso...")
        self.log_to_console("Training del modello Q-Learning...\n")

        def train_callback(episode, total):
            self.log_to_console(f"Progresso training: {episode}/{total}")

        def do_training():
            self.agent.train(self.env, num_episodes=500000, callback=train_callback)
            self.root.after(0, self.training_complete)

        thread = threading.Thread(target=do_training, daemon=True)
        thread.start()

    def training_complete(self):
        self.log_to_console("\n✓ Training completato!")
        self.log_to_console(f"Q-table: {len(self.agent.q_table)} stati")
        self.log_to_console("\nRegole: Dealer sta su 17 hard, pesca su 17 soft")
        self.log_to_console("\nIl modello sta giocando automaticamente...")
        self.log_to_console("Osserva come prende le decisioni!\n")
        self.load_images()

    def load_images(self):
        try:
            for i in range(1, 11):
                img = Image.open(f"{UPLOAD_PATH}/{i}.png")
                img = img.resize((70, 98), Image.Resampling.LANCZOS)
                self.card_images[i] = ImageTk.PhotoImage(img)

            img = Image.open(f"{UPLOAD_PATH}/ace.png")
            img = img.resize((70, 98), Image.Resampling.LANCZOS)
            self.card_images[11] = ImageTk.PhotoImage(img)

            img = Image.open(f"{UPLOAD_PATH}/card_back.png")
            img = img.resize((70, 98), Image.Resampling.LANCZOS)
            self.card_back = ImageTk.PhotoImage(img)

            img = Image.open(f"{UPLOAD_PATH}/still.png")
            img = img.resize((650, 280), Image.Resampling.LANCZOS)
            self.table_bg = ImageTk.PhotoImage(img)

            for name in ['lock', 'win', 'lose', 'draw']:
                img = Image.open(f"{UPLOAD_PATH}/{name}.png")
                img = img.resize((650, 280), Image.Resampling.LANCZOS)
                self.bg_images[name] = ImageTk.PhotoImage(img)

            self.show_lock_screen()
            # Avvia automaticamente la prima mano dopo 3 secondi
            self.root.after(3000, self.new_hand)
        except Exception as e:
            self.log_to_console(f"Errore caricamento immagini: {e}")

    def create_ui(self):
        main_frame = tk.Frame(self.root, bg='#1a472a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        game_frame = tk.Frame(main_frame, bg='#1a472a', width=650)
        game_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        game_frame.pack_propagate(False)
        
        image_frame = tk.Frame(game_frame, bg='#1a472a', height=280)
        image_frame.pack(fill=tk.X, pady=(0, 5))
        image_frame.pack_propagate(False)

        self.canvas_image = tk.Canvas(image_frame, bg='#0d5c2d', highlightthickness=0)
        self.canvas_image.pack(fill=tk.BOTH, expand=True)

        #CARTE
        cards_frame = tk.Frame(game_frame, bg='#0d5c2d', height=250)
        cards_frame.pack(fill=tk.X, pady=(5, 5))
        cards_frame.pack_propagate(False)

        self.canvas_cards = tk.Canvas(cards_frame, bg='#0d5c2d', highlightthickness=2,
                                      highlightbackground='#8B4513')
        self.canvas_cards.pack(fill=tk.BOTH, expand=True)

        #BOTTONI
        control_frame = tk.Frame(game_frame, bg='#1a472a', height=80)
        control_frame.pack(fill=tk.X, pady=(5, 0))
        control_frame.pack_propagate(False)

        btn_style = {'font': ('Arial', 14, 'bold'), 'width': 12, 'height': 2}

        self.btn_start = tk.Button(control_frame, text="NUOVA MANO",
                                   command=self.new_hand, bg='#4CAF50', fg='white', **btn_style)
        self.btn_start.pack(side=tk.LEFT, padx=8, pady=10)

        self.btn_hit = tk.Button(control_frame, text="HIT",
                                 command=self.player_hit, bg='#2196F3', fg='white',
                                 state=tk.DISABLED, **btn_style)
        self.btn_hit.pack(side=tk.LEFT, padx=8, pady=10)

        self.btn_stand = tk.Button(control_frame, text="STAND",
                                   command=self.player_stand, bg='#FF9800', fg='white',
                                   state=tk.DISABLED, **btn_style)
        self.btn_stand.pack(side=tk.LEFT, padx=8, pady=10)

        #Q-LEARNING
        console_frame = tk.Frame(main_frame, bg='#1a472a', width=400)
        console_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        console_title = tk.Label(console_frame, text="Soft17 - Q-Learning",
                                 font=('Courier', 12, 'bold'), bg='#1a472a', fg='#FFD700')
        console_title.pack(pady=(0, 10))

        self.console = scrolledtext.ScrolledText(console_frame,
                                                 font=('Courier', 9),
                                                 bg='#0a0a0a', fg='#00ff00',
                                                 wrap=tk.WORD)
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.config(state=tk.DISABLED)

    def log_to_console(self, message):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)

    def show_lock_screen(self):
        self.canvas_image.delete("all")
        self.canvas_cards.delete("all")
        if 'lock' in self.bg_images:
            self.canvas_image.create_image(325, 140,
                                           image=self.bg_images['lock'], anchor=tk.CENTER)

    def new_hand(self):
        if not self.card_images:
            messagebox.showwarning("Attenzione", "Caricamento immagini in corso...")
            return

        self.show_lock = False
        self.game_active = True
        self.dealer_revealed = False
        self.state = self.env.reset()
        self.btn_stand.config(state=tk.DISABLED)
        self.btn_hit.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)
        self.log_to_console("\n" + "=" * 50)
        self.log_to_console("NUOVA MANO INIZIATA")
        self.log_to_console("=" * 50)
        self.draw_table()
        reasoning = self.agent.get_reasoning(self.state, self.env)
        self.log_to_console("\n" + reasoning)
        
        # Avvia il gioco automatico dopo 5 secondi per osservare l'analisi
        self.root.after(5000, self.auto_play)

    def player_stand(self):
        if not self.game_active:
            return
        self.log_to_console("\n>>> GIOCATORE: STAND <<<")
        self.dealer_revealed = True
        self.draw_table()
        self.root.update()
        time.sleep(1.5)
        self.state, reward, done, info = self.env.step(self.state, 0)
        self.draw_table()
        self.root.update()
        time.sleep(1.5)
        self.log_game_result(reward, info)
        self.end_game(reward)

    def player_hit(self):
        if not self.game_active:
            return
        self.log_to_console("\n>>> GIOCATORE: HIT <<<")
        # Pesca carta
        self.state, reward, done, info = self.env.step(self.state, 1)
        self.draw_table()
        if done:
            # Giocatore sballato
            value, _ = self.env.get_hand_value(self.state['player_hand'])
            self.log_to_console(f"\nSBALLATO! Valore: {value}")
            self.dealer_revealed = True
            self.draw_table()
            self.log_game_result(reward, info)
            self.end_game(reward)
        else:
            reasoning = self.agent.get_reasoning(self.state, self.env)
            self.log_to_console("\n" + reasoning)

    def auto_play(self):
        """Metodo che fa giocare automaticamente il modello"""
        if not self.game_active:
            return
        
        # Ottieni la decisione del modello
        action = self.agent.get_best_action(self.state, self.env)
        
        if action == 0:  # STAND
            self.player_stand()
        else:  # HIT
            self.player_hit()
            # Se il gioco è ancora attivo, continua dopo 5 secondi
            if self.game_active:
                self.root.after(5000, self.auto_play)

    def log_game_result(self, reward, info):
        pv, _ = self.env.get_hand_value(self.state['player_hand'])
        dv, _ = self.env.get_hand_value(self.state['dealer_hand'])
        self.log_to_console("\nRISULTATO:")
        self.log_to_console(f"Dealer: {dv}")
        self.log_to_console(f"Giocatore: {pv}")
        if reward > 0:
            self.log_to_console("VITTORIA!")
        elif reward < 0:
            self.log_to_console("SCONFITTA")
        else:
            self.log_to_console("PAREGGIO")

    def end_game(self, reward):
        self.game_active = False
        self.dealer_revealed = True
        self.btn_stand.config(state=tk.DISABLED)
        self.btn_hit.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)
        # RISULTATO
        self.root.after(0, lambda: self.show_result(reward))
        # Avvia automaticamente una nuova mano dopo 5 secondi
        self.root.after(5000, self.new_hand)

    def show_result(self, reward):
        key = 'win' if reward > 0 else ('lose' if reward < 0 else 'draw')
        if key in self.bg_images:
            self.canvas_image.delete("all")
            self.canvas_image.create_image(325, 140,
                                           image=self.bg_images[key], anchor=tk.CENTER)

    def draw_table(self):
        if self.show_lock:
            self.show_lock_screen()
            return

        # CANVAS TAVOLO
        self.canvas_image.delete("all")
        if hasattr(self, 'table_bg'):
            self.canvas_image.create_image(325, 140,
                                           image=self.table_bg, anchor=tk.CENTER)

        # CANVAS CARTE
        self.canvas_cards.delete("all")

        # Larghezza canvas carte
        cards_width = 650
        cards_height = 250

        # AREA DEALER
        dealer_y = 60

        dealer_hand = self.state['dealer_hand']
        total_dealer_width = len(dealer_hand) * 80
        dealer_start_x = (cards_width - total_dealer_width) / 2

        for i, card in enumerate(dealer_hand):
            x = dealer_start_x + i * 80 + 35
            if i == 1 and not self.dealer_revealed:
                self.canvas_cards.create_image(x, dealer_y, image=self.card_back, anchor=tk.CENTER)
            else:
                img = self.card_images.get(card, self.card_back)
                self.canvas_cards.create_image(x, dealer_y, image=img, anchor=tk.CENTER)

        dealer_label_x = dealer_start_x + total_dealer_width + 60
        self.canvas_cards.create_text(dealer_label_x, dealer_y, text="DEALER",
                                      font=('Arial', 14, 'bold'), fill='white', anchor=tk.W)

        if self.dealer_revealed:
            dv, soft = self.env.get_hand_value(self.state['dealer_hand'])
            value_text = f"{dv}{' (soft)' if soft else ''}"
            self.canvas_cards.create_text(dealer_label_x, dealer_y + 25, text=value_text,
                                          font=('Arial', 12, 'bold'), fill='yellow', anchor=tk.W)

        # AREA PLAYER
        player_y = 175

        player_hand = self.state['player_hand']
        total_player_width = len(player_hand) * 80
        player_start_x = (cards_width - total_player_width) / 2

        for i, card in enumerate(player_hand):
            x = player_start_x + i * 80 + 35
            img = self.card_images.get(card, self.card_back)
            self.canvas_cards.create_image(x, player_y, image=img, anchor=tk.CENTER)

        player_label_x = player_start_x + total_player_width + 60
        self.canvas_cards.create_text(player_label_x, player_y, text="GIOCATORE",
                                      font=('Arial', 14, 'bold'), fill='white', anchor=tk.W)

        pv, soft = self.env.get_hand_value(self.state['player_hand'])
        value_text = f"{pv}{' (soft)' if soft else ''}"
        text_color = 'red' if pv > 21 else 'yellow'
        self.canvas_cards.create_text(player_label_x, player_y + 25, text=value_text,
                                      font=('Arial', 12, 'bold'), fill=text_color, anchor=tk.W)


def main():
    root = tk.Tk()
    game = BlackjackGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
