# https://qiita.com/katryo/items/6a2266ffafb7efa9a46c
import math
import sys
import MeCab

class NaiveBayes:
  def __init__(self):
    self.vocabularies = set()
    self.word_count = {} # {'category1': {'word1': 4, 'word2': 1, ...}}
    self.category_count = {} # {'category1': 4, 'category2': 7, ...}

  def fit(self, document, category):
    words = self.to_words(document)
    for w in words:
      self.word_count_up(w, category)

    self.category_count_up(category)

  def predict(self, document):
    # 事後確率のP(cat|doc)が一番大きいカテゴリを返す
    best = None
    max = -sys.maxsize
    words = self.to_words(document)
    for cat in self.category_count.keys():
      score = self.score(words, cat)
      print ('カテゴリ: %s => %s' % (cat, score))
      if score > max:
        max = score
        best = cat

    return best

  def to_words(self, document):
    """
    入力: 'すべて自分のほうへ'
    出力: tuple(['すべて', '自分', 'の', 'ほう', 'へ'])
    """
    tagger = MeCab.Tagger('mecabrc')  # 別のTaggerを使ってもいい
    mecab_result = tagger.parse(document)
    info_of_words = mecab_result.split('\n')
    words = []
    for info in info_of_words:
        # macabで分けると、文の最後に’’が、その手前に'EOS'が来る
        if info == 'EOS' or info == '':
            break
            # info => 'な\t助詞,終助詞,*,*,*,*,な,ナ,ナ'
        info_elems = info.split(',')
        # 6番目に、無活用系の単語が入る。もし6番目が'*'だったら0番目を入れる
        if info_elems[6] == '*':
            # info_elems[0] => 'ヴァンロッサム\t名詞'
            words.append(info_elems[0][:-3])
            continue
        words.append(info_elems[6])
    return tuple(words)

  def word_count_up(self, word, category):
    self.word_count.setdefault(category, {})
    self.word_count[category].setdefault(word, 0)
    self.word_count[category][word] += 1
    self.vocabularies.add(word)

  def category_count_up(self, category):
    self.category_count.setdefault(category, 0)
    self.category_count[category] += 1

  def score(self, word, category):
    # log(P(cat|doc))を求める問題。documentを与えられたあとにcategoryに属する確率。
    # logなのは桁数を小さくするため
    # P(cat|doc) = P(doc|cat)*P(cat) / P(doc)
    # P(doc|cat) = P(word1|cat)*P(word2|cat)*..*P(wordn|cat)
    # P(doc)はどのカテゴリでも共通なので無視
    score = math.log(self.p_cat(category))
    for w in word:
      score += math.log(self.p_word(w, category))
    return score

  def p_cat(self, category):
    # P(cat)
    return float(self.category_count[category] / sum(self.category_count.values()))

  def p_word(self, word, category):
    # P(word|cat)
    return float((self.word_count.get(category, {}).get(word, 0) + 1) / (sum(self.word_count[category].values()) + len(self.vocabularies)))

if __name__ == "__main__":
  nb = NaiveBayes()
  nb.fit('''Python（パイソン）は、オランダ人のグイド・ヴァンロッサムが作ったオープンソースのプログラミング言語。
オブジェクト指向スクリプト言語の一種であり、Perlとともに欧米で広く普及している。イギリスのテレビ局 BBC が製作したコメディ番組『空飛ぶモンティパイソン』にちなんで名付けられた。
Pythonは英語で爬虫類のニシキヘビの意味で、Python言語のマスコットやアイコンとして使われることがある。Pythonは汎用の高水準言語である。プログラマの生産性とコードの信頼性を重視して設計されており、核となるシンタックスおよびセマンティクスは必要最小限に抑えられている反面、利便性の高い大規模な標準ライブラリを備えている。
Unicodeによる文字列操作をサポートしており、日本語処理も標準で可能である。 多くのプラットフォームをサポートしており（動作するプラットフォーム）、また、豊富なドキュメント、豊富なライブラリがあることから、産業界でも利用が増えつつある。''', 'Python')

  nb.fit('''Ruby（ルビー）は、まつもとゆきひろ（通称Matz）により開発されたオブジェクト指向スクリプト言語であり、従来Perlなどのスクリプト言語が用いられてきた領域でのオブジェクト指向プログラミングを実現する。Rubyは当初1993年2月24日に生まれ、1995年12月にfj上で発表された。名称のRubyは、プログラミング言語Perlが6月の誕生石であるPearl（真珠）と同じ発音をすることから、まつもとの同僚の誕生石（7月）のルビーを取って名付けられた。''', 'Ruby')

  nb.fit('''豊富な機械学習（きかいがくしゅう、Machine learning）とは、人工知能における研究課題の一つで、人間が自然に行っている学習能力と同様の機能をコンピュータで実現させるための技術・手法のことである。 ある程度の数のサンプルデータ集合を対象に解析を行い、そのデータから有用な規則、ルール、知識表現、判断基準などを抽出する。 データ集合を解析するため、統計学との関連も非常に深い。
機械学習は検索エンジン、医療診断、スパムメールの検出、金融市場の予測、DNA配列の分類、音声認識や文字認識などのパターン認識、ゲーム戦略、ロボット、など幅広い分野で用いられている。応用分野の特性に応じて学習手法も適切に選択する必要があり、様々な手法が提案されている。それらの手法は、Machine Learning や IEEE Transactions on Pattern Analysis and Machine Intelligence などの学術雑誌などで発表されることが多い。''', '機械学習')
  nb.fit('''ルビー（英: Ruby、紅玉）は、コランダム（鋼玉）の変種である。赤色が特徴的な宝石である。
                天然ルビーは産地がアジアに偏っていて欧米では採れないうえに、
                産地においても宝石にできる美しい石が採れる場所は極めて限定されており、
                3カラットを超える大きな石は産出量も少ない。
             ''', 'Gem')
  nb.fit('''ヴァンロッサム氏''', 'Python')
  nb.fit('''ヴァンロッサム氏''', 'Python')
  nb.fit('''ヴァンロッサム氏''', 'Python')
  nb.fit('''ヴァンロッサム氏''', 'Python')
  nb.fit('''豊富な機械学習（きかいがくしゅう、Machine learning）とは、人工知能における研究課題の一つで、人間が自然に行っている学習能力と同様の機能をコンピュータで実現させるための技術・手法のことである。 ある程度の数のサンプルデータ集合を対象に解析を行い、そのデータから有用な規則、ルール、知識表現、判断基準などを抽出する。 データ集合を解析するため、統計学との関連も非常に深い。
機械学習は検索エンジン、医療診断、スパムメールの検出、金融市場の予測、DNA配列の分類、音声認識や文字認識などのパターン認識、ゲーム戦略、ロボット、など幅広い分野で用いられている。応用分野の特性に応じて学習手法も適切に選択する必要があり、様々な手法が提案されている。それらの手法は、Machine Learning や IEEE Transactions on Pattern Analysis and Machine Intelligence などの学術雑誌などで発表されることが多い。''', '機械学習')
  nb.fit('''豊富な機械学習（きかいがくしゅう、Machine learning）とは、人工知能における研究課題の一つで、人間が自然に行っている学習能力と同様の機能をコンピュータで実現させるための技術・手法のことである。 ある程度の数のサンプルデータ集合を対象に解析を行い、そのデータから有用な規則、ルール、知識表現、判断基準などを抽出する。 データ集合を解析するため、統計学との関連も非常に深い。
機械学習は検索エンジン、医療診断、スパムメールの検出、金融市場の予測、DNA配列の分類、音声認識や文字認識などのパターン認識、ゲーム戦略、ロボット、など幅広い分野で用いられている。応用分野の特性に応じて学習手法も適切に選択する必要があり、様々な手法が提案されている。それらの手法は、Machine Learning や IEEE Transactions on Pattern Analysis and Machine Intelligence などの学術雑誌などで発表されることが多い。''', '機械学習')
  for i in range(100):
    nb.fit('''豊富な機械学習''', '機械学習')

  words = 'ヴァンロッサム氏によって開発されました.'
  print ('%s => 推定カテゴリ: %s' % (words , nb.predict(words)))

  words = 'スパムメールがきた'
  print ('%s => 推定カテゴリ: %s' % (words , nb.predict(words)))

  words = '「機械学習 はじめよう」が始まりました.'
  print ('%s => 推定カテゴリ: %s' % (words , nb.predict(words)))
