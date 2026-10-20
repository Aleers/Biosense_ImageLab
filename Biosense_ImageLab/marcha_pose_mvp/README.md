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

Depois da deteccao, revise e calibre o video. As correcoes sao rastreadas ate o fim; perdas sao sinalizadas e nao entram nas medidas. A interface aplica o resultado completo somente apos sucesso.

## Fases e parâmetros da marcha

No modo lateral, verde indica apoio associado a uma pisada registrada; amarelo indica apoio sem novo contato confirmado e laranja indica balanco. A classificacao continua sendo uma heuristica 2D, com confirmacao temporal.

Com calibração, cada contato inicial atualiza o comprimento do passo (distância entre tornozelos) e da passada (deslocamento entre contatos consecutivos do mesmo calcanhar). A câmera deve estar fixa, perpendicular e a referência de calibração deve estar no plano da marcha.

O vídeo também exibe uma tabela ampliada dos ângulos com valor atual, mínimo e máximo. À direita, um painel no mesmo padrão mostra as fases coloridas e listas acumuladas de `Pisada 01`, `Pisada 02`... e `Passada 01`, `Passada 02`..., com lado e distância. O espaçamento se adapta à quantidade detectada.

Contatos sao confirmados por persistencia temporal e alternancia. A contagem independe da calibracao e de distancia minima de 5 cm. O apoio ja presente ao iniciar o video nao e contado como contato novo. Candidatos rejeitados ficam registrados na sessao.

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

O rastro da orelha e as correcoes acompanhadas continuam ate o fim do video. Os criterios de pisada ficam em GaitOptions: confirmacao de 0.06 s (minimo dois quadros), intervalo de 0.30 s na mesma perna e 0.10 s entre lados alternados.

Para calibrar, selecione **Calibrar**, clique nas duas extremidades de um objeto visível com tamanho conhecido e informe a distância em centímetros. A escala é usada nas medidas lineares do CSV. Os ângulos não dependem dessa escala.

No modo lateral, o rastro acumulado das orelhas permanece visivel. Com calibracao, o CSV inclui distancias entre tornozelos, deslocamento da orelha e caminho acumulado.

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


## Qualidade dos dados, rastreamento e sessoes

`LandmarkValidator` centraliza validacao de coordenadas finitas, visibilidade,
indices e calibracao. Os mesmos criterios sao aplicados a angulos, medidas
entre landmarks, metricas frontais e eventos da marcha. Pontos indisponiveis
nao geram medidas; lacunas nao geram eventos de contato/retirada.
As formulas angulares e as colunas legadas do CSV foram preservadas.

O rastreamento de correcoes fica em `tracking.py`. CSRT e candidatos brancos
sao verificados por aparencia, continuidade e ambiguidade. Os estados registrados
sao `manual`, `tracking`, `suspect`, `lost` e `reacquired`. A reaquisi??o exige
confirmacao em quadros consecutivos. Em perda/ambiguidade, as coordenadas automaticas
ficam preservadas, mas o ponto corrigido recebe visibilidade zero e nao entra nas
medidas ate ser confirmado novamente. O valor tecnico 0.8 de aceita??o do rastreador
nao representa uma probabilidade calibrada de acerto. A visibilidade original do
MediaPipe permanece na camada original da sessao.

A thread de rastreamento produz um lote isolado. A interface aplica o lote inteiro
somente apos sucesso. Falhas de leitura nao aplicam correcoes parciais. Fechar a
revisao cancela o trabalho pendente. A publicacao de MP4/CSV usa arquivos temporarios
e restaura arquivos anteriores se uma substituicao falhar; nao e uma transacao
resistente a queda de energia entre arquivos.

Uma sessao `<nome>_<id>.session.zip` e salva ao terminar a deteccao, apos alteracoes
da revisao e ao exportar. Para reabrir, use **Selecionar** na entrada e escolha o
filtro **Sessoes**. O video original precisa continuar no caminho registrado.
Se ele foi movido, a API `SessionStore.load(path, input_path=novo_caminho)` permite
informar sua nova localizacao; o conteudo e verificado por SHA-256.

O arquivo de sessao inclui:

- `landmarks.npz`: landmarks originais antes da estabilizacao/correcoes e atuais,
  com mascara explicita de quadros sem pessoa; sem pickle;
- `manifest.json`: identificador, configuracoes, calibracao, medidas, historico
  de desfazer, eventos de auditoria, estados e fontes do rastreamento;
- hashes do video e do codigo, versoes das dependencias e timestamps nominais;
- metricas exportadas em precisao completa, antes do arredondamento do CSV.

O CSV continua com no maximo uma casa decimal, inclusive `time_s`. A sessao preserva
a precisao completa. Os timestamps atuais usam `frame / fps`, nao timestamps reais
do container; videos com taxa variavel exigem validacao especifica. O campo legado
`*_knee_foot_offset_cm` ainda contem pixels quando nao ha calibracao; a unidade real
fica registrada no manifesto. A convencao de inclinacao pelvica foi preservada.
Os dados "originais" sao a saida do MediaPipe com sua suavizacao interna ativada.

Filtros adicionais e fluxo optico sao opcionais e desativados por padrao:

```python
from processor import ProcessingOptions
options = ProcessingOptions(
    smoothing_seconds=0.04,
    tracking_use_optical_flow=True,
    tracking_smoothing_seconds=0.0,
)
```

A suavizacao temporal usa uma constante em segundos e reinicia apos lacunas ou
pontos de baixa qualidade. O fluxo optico verifica o erro de ida/volta e a aparencia
do alvo antes de aceitar uma posicao. Testes sinteticos mostram reducao de ruido
estacionario e recuperacao de um alvo com brilho abaixo do limiar branco, mas nao
comprovam melhora em videos reais. Parametros de movimento/aparencia sao heuristicos.
A ausencia de CSRT e registrada; a busca de marcadores continua disponivel.

Para executar os testes sem pytest:

```powershell
python -B -m unittest -v test_joint_angles test_robustness test_gait_display
```

A sessao possui dados identificaveis e caminhos locais; armazene-a com os mesmos
controles usados para os videos do estudo. O historico tecnico nao e uma trilha de
auditoria assinada, autenticada por usuario ou protegida contra alteracao deliberada.


## Registro de pisadas e continuidade visual

`GaitPhaseAnalyzer` confirma fases em uma janela expressa em segundos e aceita
calcanhar ou ponta do pe visivel. `FootfallRegistry` separa contato de medida:
um contato alternado confirmado recebe numero mesmo sem calibracao ou sem os
dois tornozelos visiveis. A distancia pode ser obtida nos proximos 0.20 s de apoio;
`measurement_frame` registra o quadro usado. Valores abaixo de 5 cm nao removem
contatos. Uma repeticao do mesmo lado e sinalizada como `nonalternating_contact`,
sem bloquear o proximo contato valido do lado oposto. Apoio inicial ou retorno
apos uma lacuna longa nao sao convertidos automaticamente em contato novo.

A sessao guarda `gait_records` com decisoes e distancias, e `display_records`
com a qualidade da apresentacao. `software_identity` preserva a identificacao
da deteccao original; `processing_identity` registra o codigo atual no salvamento.

`VisualContinuity` atua somente na imagem, na revisao e na exportacao. A falta
breve de visibilidade mantem o ultimo angulo por ate 0.20 s, indicado por `~`.
Depois desse prazo, o rotulo permanece com `--`; o esqueleto anterior fica como
referencia tracejada cinza. Referencias retidas nao representam a posicao atual
observada e nao entram no CSV, nas distancias ou na deteccao de pisadas.
Valores retidos tambem nao atualizam minimos e maximos. Os textos sao limitados
as bordas do quadro e desenhados apos os paineis para evitar ocultacao.

A exportacao de verificacao do video real fica em `validation_outputs/`.
A contagem gerada e algoritmica: revisar o video e comparar com anotacoes de
referencia continua necessario para estimar sensibilidade e falsos positivos.
