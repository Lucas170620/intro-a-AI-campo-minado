from game.campo_minado import CampoMinado

if __name__ == "__main__":
    try:
        linhas = int(input("Digite o número de linhas do campo: "))
        colunas = int(input("Digite o número de colunas do campo: "))
        num_bombas = int(input("Digite o número de bombas: "))

        if num_bombas >= linhas * colunas:
            print("Número de bombas inválido! Deve ser menor que o total de células.")
        else:
            jogo = CampoMinado(linhas, colunas, num_bombas)

            while jogo.jogo_ativo:
                jogo.mostrar_campo()
                try:
                    l = int(input("Linha: "))
                    c = int(input("Coluna: "))
                    jogo.revelar(l, c)
                except ValueError:
                    print("Digite um número válido!")
    except ValueError:
        print("Por favor, insira apenas números inteiros!")