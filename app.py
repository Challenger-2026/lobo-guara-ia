from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
app.json.ensure_ascii = False
app.json.sort_keys = False
CORS(app, origins=['https://lobo-guara-swart.vercel.app', 'http://localhost:5173'])

# Modelos treinados no notebook
modelo = joblib.load('modelo_risco_abandono.joblib')
perfis = joblib.load('modelo_perfis.joblib')
usuarios_teste = pd.read_csv('usuarios_teste.csv')

ATRIBUTOS = ['total_acoes', 'streak_maximo', 'diversidade_categorias', 'pontos_totais',
             'interacoes_sociais_media', 'posicao_ranking_media', 'tempo_medio_entre_acoes_dias']

# Mesmo tratamento dos ausentes do notebook
PREENCHIMENTO = {'posicao_ranking_media': 213, 'tempo_medio_entre_acoes_dias': 30}

ORIENTACOES = {
    'interacoes_sociais_media': 'Postar e entrar em comunidades',
    'streak_maximo': 'Meta de sequência de 7 dias',
    'diversidade_categorias': 'Missão de uma categoria nova',
    'tempo_medio_entre_acoes_dias': 'Notificação de lembrete',
    'posicao_ranking_media': 'Liga semanal e quanto falta para subir',
}

TOM = {
    'Inativos': 'Reativação: convidar o usuário a voltar com uma ação simples.',
    'Intermediários': 'Desafio: propor metas curtas para criar o hábito.',
    'Engajados': 'Reconhecimento: valorizar a constância e manter o ritmo.',
}


def gerar_orientacoes(usuario):
    # Peso de cada atributo no risco = coeficiente x valor padronizado
    padronizacao = modelo.named_steps['padronizacao']
    classificador = modelo.named_steps['classificador']
    valores_padronizados = padronizacao.transform(usuario)[0]
    pesos = classificador.coef_[0] * valores_padronizados

    contribuicoes = pd.Series(pesos, index=ATRIBUTOS)
    contribuicoes = contribuicoes[list(ORIENTACOES.keys())]
    principais = contribuicoes[contribuicoes > 0].sort_values(ascending=False).head(2)

    return [{'atributo': atributo, 'acao': ORIENTACOES[atributo]} for atributo in principais.index]


@app.route('/')
def inicio():
    return PAGINA


@app.route('/sortear')
def sortear():
    usuario = usuarios_teste.sample(1).iloc[0]
    resposta = {'usuario_id': usuario['usuario_id'],
                'rotulo_real': usuario['engajamento_30dias']}
    for atributo in ATRIBUTOS:
        resposta[atributo] = float(usuario[atributo])
    return jsonify(resposta)


@app.route('/prever', methods=['POST'])
def prever():
    dados = request.get_json(silent=True)
    if dados is None:
        return jsonify({'erro': 'Envie os dados do usuário em JSON.'}), 400

    faltando = [atributo for atributo in ATRIBUTOS if atributo not in dados]
    if faltando:
        return jsonify({'erro': 'Campos obrigatórios ausentes.', 'campos': faltando}), 400

    valores = {}
    for atributo in ATRIBUTOS:
        valor = dados[atributo]
        if valor is None or valor == '':
            valor = PREENCHIMENTO.get(atributo, 0)
        try:
            valores[atributo] = float(valor)
        except (TypeError, ValueError):
            return jsonify({'erro': f'Valor inválido em {atributo}.'}), 400

    usuario = pd.DataFrame([valores], columns=ATRIBUTOS)

    classe = modelo.predict(usuario)[0]
    probabilidades = dict(zip(modelo.classes_, modelo.predict_proba(usuario)[0]))
    grupo = int(perfis['modelo'].predict(usuario)[0])
    perfil = perfis['nomes'][grupo]

    return jsonify({
        'classe': classe,
        'probabilidade_risco': round(float(probabilidades['risco_abandono']), 2),
        'perfil': perfil,
        'tom': TOM[perfil],
        'orientacoes': gerar_orientacoes(usuario),
        'valores_usados': valores,
    })


PAGINA = '''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lobo-guará Tech – API de Risco de Abandono</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 720px; margin: 24px auto; padding: 0 16px; color: #222; }
  h1 { font-size: 22px; }
  form { display: grid; grid-template-columns: 1fr 140px; gap: 8px 12px; align-items: center; }
  input { padding: 6px; }
  button { margin: 16px 8px 0 0; padding: 8px 16px; cursor: pointer; }
  pre { background: #f4f4f4; padding: 12px; overflow-x: auto; }
</style>
</head>
<body>
<h1>Lobo-guará Tech – Previsão de Risco de Abandono</h1>
<p>Sorteie um usuário de teste (que o modelo não viu no treino), altere os valores se quiser e clique em Prever.</p>
<form id="formulario"></form>
<button id="sortear">Sortear usuário</button>
<button id="prever">Prever</button>
<h2>Resposta</h2>
<pre id="resposta">-</pre>
<script>
  const atributos = ['total_acoes', 'streak_maximo', 'diversidade_categorias', 'pontos_totais',
                     'interacoes_sociais_media', 'posicao_ranking_media', 'tempo_medio_entre_acoes_dias'];
  const formulario = document.getElementById('formulario');
  const resposta = document.getElementById('resposta');

  atributos.forEach(a => {
    formulario.innerHTML += `<label for="${a}">${a}</label><input id="${a}" type="number" step="any">`;
  });

  document.getElementById('sortear').onclick = async () => {
    const r = await fetch('/sortear');
    const usuario = await r.json();
    atributos.forEach(a => document.getElementById(a).value = usuario[a]);
    resposta.textContent = JSON.stringify(usuario, null, 2);
  };

  document.getElementById('prever').onclick = async () => {
    const dados = {};
    atributos.forEach(a => {
      const valor = document.getElementById(a).value;
      dados[a] = valor === '' ? null : Number(valor);
    });
    const r = await fetch('/prever', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(dados)
    });
    resposta.textContent = JSON.stringify(await r.json(), null, 2);
  };
</script>
</body>
</html>'''


if __name__ == '__main__':
    app.run(debug=True)
