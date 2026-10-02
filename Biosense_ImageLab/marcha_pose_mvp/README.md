# Análise automática de marcha — MVP

Aplicativo desktop que recebe um vídeo frontal ou lateral, detecta automaticamente os pontos corporais e exporta:

- vídeo MP4 com esqueleto, rastreio e ângulos;
- CSV com ângulos de quadril, joelho e tornozelo em cada quadro;
- seleção automática do lado mais visível ou análise dos dois lados.
- revisão quadro a quadro com correção manual dos pontos;
- reprodução do resultado antes da exportação;
- calibração por dois pontos e uma distância real conhecida;

## Organizacao do codigo para o TCC

O projeto foi organizado em torno de objetos com responsabilidades definidas:

- `MarchaApp`: controla a janela inicial e o fluxo entre as etapas;
- `ProcessingOptions`: representa as configuracoes imutaveis da analise;
- `Analysis`: concentra resultados e alteracoes realizadas na revisao;
- `PoseVideoAnalyzer`: le o video e executa o MediaPipe quadro a quadro;
- `ReviewWindow`: apresenta os resultados e recebe as correcoes do usuario;
- `VideoExporter`: coordena a geracao do MP4 e do CSV;
- `Translator`: seleciona o idioma e fornece textos compativeis com o OpenCV.

As funcoes `analyze_video` e `render_video` foram mantidas como fachadas de
compatibilidade. A interface permanece simples e cada etapa principal pode ser
explicada, testada ou substituida separadamente.

## Instalação

Use Python 3.10 ou 3.11. No Windows, abra o PowerShell dentro da pasta do projeto:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Se estiver atualizando uma instalação de uma versão anterior, execute antes:

```powershell
pip uninstall -y opencv-python opencv-contrib-python
pip install -r requirements.txt
```

Esta versão usa `opencv-contrib-python` porque o rastreador CSRT não faz parte do pacote OpenCV básico.

No Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## Como gravar

1. Apoie a câmera; não grave com ela na mão.
2. Mostre o corpo inteiro, incluindo os pés, durante todo o vídeo.
3. Para a vista lateral, deixe a câmera perpendicular ao sentido da marcha e aproximadamente na altura do quadril.
4. Para a vista frontal, centralize o trajeto e evite perspectiva diagonal.
5. Use iluminação uniforme, roupa que contraste com o fundo e apenas uma pessoa no quadro.
6. Se possível, grave a 60 fps; 30 fps funciona para o protótipo.

## Definição dos ângulos

- Quadril: ombro–quadril–joelho.
- Joelho: quadril–joelho–tornozelo.
- Tornozelo: joelho–tornozelo–ponta do pé (calcanhar como alternativa).

Os valores são ângulos internos 2D entre 0° e 180°. O detector do vídeo aplica suavização temporal aos pontos antes da etapa de revisão.

## Revisão e calibração

Depois da detecção, o aplicativo abre a tela de revisão. Arraste a barra para navegar no vídeo ou use **Reproduzir/Pausar**. Para corrigir, clique em um ponto verde e depois na posição correta. Nos 180 quadros seguintes, um rastreador CSRT segue a aparência da bolinha e uma máscara branca refina o seu centro. Quando ambos perdem a referência, o programa utiliza a estimativa automática com influência decrescente da correção.

## Fases e parâmetros da marcha

No modo lateral, o pé fica verde durante **APOIO** e laranja durante **BALANÇO**. As transições são marcadas como **CONTATO INICIAL** e **RETIRADA DO PÉ**. A classificação é uma heurística baseada na proximidade estimada do solo e na velocidade do pé; os 60%/40% são referências esperadas, não valores impostos pelo programa.

Com calibração, cada contato inicial atualiza o comprimento do passo (distância entre tornozelos) e da passada (deslocamento entre contatos consecutivos do mesmo calcanhar). A câmera deve estar fixa, perpendicular e a referência de calibração deve estar no plano da marcha.

O vídeo também exibe uma tabela ampliada dos ângulos com valor atual, mínimo e máximo. À direita, um painel no mesmo padrão mostra as fases coloridas e listas acumuladas de `Pisada 01`, `Pisada 02`... e `Passada 01`, `Passada 02`..., com lado e distância. O espaçamento se adapta à quantidade detectada.

Eventos separados por menos de 0,35 s na mesma perna e distâncias inferiores a 5 cm são ignorados. Cada pisada aceita deixa uma marca numerada na posição do calcanhar no vídeo.

Na vista frontal, a tabela esquerda inclui os ângulos Q estimados de ambos os lados. O painel direito **STEP DOWN 2D** apresenta valor atual, mínimo e máximo de FPPA, inclinação da pelve, inclinação do tronco e deslocamento horizontal joelho–pé. Sem calibração, o deslocamento é mostrado em pixels; com calibração, em centímetros. São medidas cinemáticas 2D auxiliares, não uma classificação clínica automática.

## Idiomas e medições manuais

A tela inicial permite selecionar Portugues, English ou Espanol; Portugues e o padrao. A selecao traduz a aplicacao e os textos desenhados no video exportado.

O nome exportado recebe o idioma: `_pt.mp4`, `_en.mp4` ou `_esp.mp4`; o CSV usa o mesmo sufixo. Os textos sao normalizados para ASCII: letras acentuadas perdem o acento e `c-cedilha` e convertido para `c`. O OpenCV desenha diretamente no frame, sem conversoes para PIL durante a exportacao.

Depois da calibração, o botão **Adicionar medida** é habilitado na revisão. Clique no botão, selecione duas posições e informe um nome. É possível criar quantos segmentos forem necessários. Eles são desenhados no vídeo e resumidos em centímetros numa tabela inferior esquerda, que só aparece quando há medidas.

O botão **Medir pontos corporais** permite escolher duas bolinhas do MediaPipe. A medida acompanha esses dois landmarks quadro a quadro, atualizando o segmento e a distância durante o vídeo. As medidas livres continuam fixas como anteriormente.

Ao concluir qualquer uma das duas medições, uma janela apresenta nove cores reutilizáveis. A cor escolhida é aplicada aos pontos, ao segmento, ao rótulo e à linha da tabela.

O botão **Desfazer** mantém o histórico da tela de revisão. Ele remove, em ordem inversa, medições livres, medições rastreadas, calibrações e edições de landmarks. Ao desfazer uma edição, todas as posições propagadas pelo rastreador também são restauradas.

## Otimizações de desempenho

- textos ASCII sao desenhados diretamente pelo OpenCV, sem conversao para PIL;
- fundos semitransparentes processam apenas a área de cada painel;
- o rastro lateral usa uma camada cumulativa e acrescenta somente o novo trecho;
- a revisão calcula os centros das orelhas uma vez e usa uma única chamada de polilinha para o trajeto.

Na análise lateral, o rastro da orelha e o rastreamento iniciado por uma correção manual continuam até o fim do vídeo. Pisadas aceitas precisam alternar os lados, respeitar intervalo mínimo e estar separadas ao menos 5 cm da posição anterior.

Para calibrar, selecione **Calibrar**, clique nas duas extremidades de um objeto visível com tamanho conhecido e informe a distância em centímetros. A escala é usada nas medidas lineares do CSV. Os ângulos não dependem dessa escala.

No modo lateral, o vídeo mostra o centro das orelhas e o rastro dos últimos 180 quadros. Com calibração, o CSV inclui distância direta e horizontal entre tornozelos, deslocamento da orelha desde o início e caminho acumulado da orelha.

## Ângulo Q estimado

Na vista frontal, o programa exporta `left_q_estimated_deg` e `right_q_estimated_deg`. É o suplemento do ângulo quadril–joelho–tornozelo, usado como aproximação 2D do alinhamento frontal. Não é o ângulo Q clínico anatômico exato: este exige EIAS, centro da patela e tuberosidade da tíbia, pontos que precisam de marcação específica e validação profissional.

## Limitações importantes

Este MVP usa o MediaPipe Pose para estimativa sem marcadores. A GPU NVIDIA não é necessária nesta versão; o processamento é executado principalmente na CPU. Uma evolução voltada à GPU pode trocar o detector por RTMPose/MMPose, mantendo a interface e a etapa de exportação.

O resultado depende do posicionamento da câmera, oclusões, roupas, iluminação e perspectiva. O sistema não mede rotação fora do plano, força, pressão plantar ou parâmetros clínicos validados. Não é dispositivo médico e não deve produzir diagnóstico ou substituir a avaliação de fisioterapeuta.

Aviso exibido pela aplicacao: prototipo de analise cinematica 2D. Nao e dispositivo medico e nao substitui avaliacao de fisioterapeuta. A camera deve permanecer fixa e perpendicular ao plano do movimento.

## Próximas evoluções recomendadas

- calibração espacial e correção de perspectiva;
- detecção automática de ciclos da marcha e eventos de contato/retirada do pé;
- gráficos de ângulo por ciclo normalizado de 0% a 100%;
- comparação entre lados e relatório;
- validação contra marcações feitas por fisioterapeutas e sistema de referência.
