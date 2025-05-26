from ai.trainer import CampoMinadoTrainer

if __name__ == "__main__":
    # Configurações do jogo
    LINHAS = 8
    COLUNAS = 8
    NUM_BOMBAS = 10
    
    # Criar e treinar o agente
    trainer = CampoMinadoTrainer(LINHAS, COLUNAS, NUM_BOMBAS)
    
    # Modo de treinamento
    print("Iniciando treinamento...")
    trainer.train(1000)  # Treina por 1000 episódios
    
    # Modo de execução
    print("\nJogando uma partida com o agente treinado...")
    trainer.play_game(show_steps=True)