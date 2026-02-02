# Soft17  - An AI Agent for BlackJack
Soft17 è un progetto per il corso di Fondamenti di Intelligenza Artificiale @ Università degli Studi di Salerno.
L'obiettivo è quello di realizzare un agente che sia in grado di giocare in modo efficiente ed automatico a Blackjack.

# Organizzazione Repository
Nella directory principale del progetto sono presenti: documentazione di progetto, presentazione e dataset (per le prime due pipeline).
Oltre ciò sono presenti diverse cartelle:
- <b>img:</b> contente le immagini che abbiamo usato all'interno della documentazione;
- <b>code</b>: contente la parte relativa al codice, essa contiene a sua volta le successive due cartelle;
- <b>Notebooks:</b> contenente i vari notebook usati durante lo sviluppo del progetto;
- <b>Demo:</b> contenente le due demo che abbiamo realizzato;

# Installazione progetto

## Utilizzo notebook
### Requisiti
Per l'utilizzo delle prime due pipeline (first, second) è necessario scaricare non solo i relativi file dalla cartella Notebooks, ma anche il dataset dalla cartella principale.
Una volta fatto ciò basta importare su Google Colab il file, selezionare il menù a tendina posto sulla sinistra, andare alla voce file, caricare il dataset nel runtime tramite il tasto "Carica in spazio di archiviazione della sessione" e poi far partire l'esecuzione tramite l'apposito tasto.
Per quanto concerne gli altri notebook basterà avviare i singoli file dopo averli scaricati ed importati.

### Replicare i risultati ottenuti
Per fare ciò basterà eseguire i singoli file senza alterarne i parametri.

## Utilizzo Demo
### Requisiti
Per quanto riguarda le demo, è necessario avere:
- Python 3.10+ installato;
- pip aggiornato (in caso contrario è sufficiente eseguire python -m pip install --upgrade pip);
- PIL, il Python Imaging Library ( in caso contrario è sufficiente eseguire pip install pillow).
### Uso della demo
Per poter utilizzare la demo è sufficiente rispettare i requisti e scaricare la cartella <b>Demo</b>.
In alternativa è possibile scaricare il singolo file di demo che si vuole utilizzare, a patto che venga scaricata anche la cartella <b>pics</b> (contenente tutte le immagini usate nella demo) e posta nella stessa directory della demo.
Una volta fatto ciò si può normalmente avviare il singolo file della demo.
Alla partenza, il modello SARSA inizia a allenarsi automaticamente.

### Caratteristiche demo
Ad ogni avvio la demo eseguirà il training del modello.
Lo stato del training verrà mostrato nella console integrata.
Una volta completato il training, si deve fare click su “NUOVA MANO” per iniziare a giocare.
Nel caso in cui si stia utilizzando la demo dell'algoritmo SARSA, allora nella console sarà riportata l'azione consigliata dal modello, ma tramite i pulsanti <b>HIT e STAND</b>
l'utente può selezionare l'azione che desidera intraprendere.
Nel caso invece della demo dell'algoritmo Q-Learning, una volta premuto "NUOVA MANO" il modello eseguirà automaticamente le singole azioni.
La console mostrerà il ragionamento dell’AI e la situazione attuale step-by-step.
