import sys
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ced_training.corpus import article_title, complete_articles, loss_mask, chunk_spans


class CorpusTests(unittest.TestCase):
    def test_adjacent_chunks_share_context_but_not_targets(self):
        counts = [3, 3, 2]
        self.assertEqual(chunk_spans(counts,0,4), [(0,0,3),(1,0,1)])
        self.assertEqual(chunk_spans(counts,4,4), [(1,1,3),(2,0,2)])
        all_targets = []
        for start in (0,4):
            for row,first,end in chunk_spans(counts,start,4):
                all_targets += [(row,i) for i in range(first,end)]
        self.assertEqual(len(all_targets),len(set(all_targets)))
        with self.assertRaises(ValueError): chunk_spans(counts,7,2)

    def test_first_million_is_exact(self):
        spans = chunk_spans([255]*39215+[175],0,1_000_000)
        self.assertEqual(len(spans),3922)
        self.assertEqual(spans[-1],(3921,0,145))
        self.assertEqual(sum(end-first for _,first,end in spans),1_000_000)

    def test_only_top_level_titles(self):
        self.assertEqual(article_title(' = Some Article = \n'), 'Some Article')
        for text in (' = = Section = = \n', 'paragraph with = sign', '', ' == Section == '):
            self.assertIsNone(article_title(text))

    def test_partial_boundary_article_is_dropped(self):
        rows = list(enumerate(['old article tail', '= = Old section = =', '', '= New =', 'body', '= = Sub = =', '= Next =', 'last']))
        articles = list(complete_articles(rows))
        self.assertEqual([a['title'] for a in articles], ['New', 'Next'])
        self.assertEqual(articles[0]['texts'], ['= New =', 'body', '= = Sub = ='])
        self.assertEqual(articles[0]['first_row'], 3)
        self.assertEqual(articles[0]['last_row'], 5)

    def test_exact_budget_and_no_cross_window_label(self):
        budget = 10_000_000
        total, windows = 0, 0
        while total < budget:
            mask = loss_mask(budget - total)
            self.assertEqual(len(mask), 256)
            self.assertEqual(mask[-1], 0)
            total += sum(mask)
            windows += 1
        self.assertEqual(total, budget)
        self.assertEqual(windows, 39216)
        self.assertEqual(sum(mask), 175)
        with self.assertRaises(ValueError):
            loss_mask(0)


if __name__ == '__main__':
    unittest.main()
