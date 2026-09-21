"""
Automated QA Test Suite for Plagiarism Detector
"""
import io
import unittest
from app import app, init_db, count_words, calculate_similarity, generate_highlighted_comparison

class PlagiarismDetectorQATestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        with app.app_context():
            init_db()

    def test_01_home_route(self):
        """Verify home page loads successfully with 200 OK."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'VeriText', response.data)
        self.assertIn(b'Detect Text Plagiarism', response.data)

    def test_02_checker_page_get(self):
        """Verify checker page loads with form inputs and sample presets."""
        response = self.client.get('/checker')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'doc1_text', response.data)
        self.assertIn(b'doc2_text', response.data)
        self.assertIn(b'Exact Copy', response.data)

    def test_03_checker_validation_empty_or_short(self):
        """Verify validation prevents checking documents under 3 words."""
        # Completely empty
        res1 = self.client.post('/checker', data={'doc1_text': '', 'doc2_text': ''})
        self.assertEqual(res1.status_code, 200)
        self.assertIn(b'Please provide text', res1.data)

        # Doc 1 too short
        res2 = self.client.post('/checker', data={'doc1_text': 'hi there', 'doc2_text': 'this is a valid long text'})
        self.assertEqual(res2.status_code, 200)
        self.assertIn(b'Document 1 is too short', res2.data)

    def test_04_successful_check_and_result_redirection(self):
        """Verify checking two valid documents redirects to /result/<id>."""
        doc1 = "Artificial intelligence enables machines to learn patterns from datasets without explicit rules."
        doc2 = "Artificial intelligence enables machines to learn complex patterns directly from training data."
        response = self.client.post('/checker', data={
            'doc1_text': doc1,
            'doc2_text': doc2
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Similarity', response.data)
        self.assertIn(b'Side-by-Side Document Highlight Inspector', response.data)
        self.assertIn(b'text-match', response.data)

    def test_05_transposed_sentences_highlighting(self):
        """Verify transposed sentences are recognized by multi-pass highlighter."""
        s1 = "Sentence alpha is here."
        s2 = "Sentence beta is there."
        doc1 = f"{s1} {s2}"
        doc2 = f"{s2} {s1}"

        h1, h2, m_words, u_words = generate_highlighted_comparison(doc1, doc2)
        # All 8 words exist in both documents
        self.assertEqual(m_words, 8)
        self.assertEqual(u_words, 0)
        self.assertIn('text-match', h1)
        self.assertIn('text-match', h2)
        self.assertNotIn('text-unique-1', h1)
        self.assertNotIn('text-unique-2', h2)

    def test_06_exact_duplicate_similarity(self):
        """Verify identical documents return 100% similarity and 0% difference."""
        text = "Deep neural networks are composed of multiple layers of nonlinear processing elements."
        sim = calculate_similarity(text, text)
        self.assertEqual(sim, 100.0)

    def test_07_word_count_consistency(self):
        """Verify word counter handles contractions and hyphens cleanly."""
        text = "Don't stop believing, state-of-the-art AI doesn't fail."
        cnt = count_words(text)
        self.assertGreaterEqual(cnt, 6)

    def test_08_file_upload_with_bom(self):
        """Verify UTF-8 file with BOM is decoded cleanly without corrupting first word."""
        content = "\ufeffMachine learning systems optimize objective loss functions.".encode('utf-8')
        doc1_file = (io.BytesIO(content), 'test.txt')
        response = self.client.post('/checker', data={
            'doc1_file': doc1_file,
            'doc2_text': 'Machine learning systems optimize loss metrics directly.'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Plagiarism analysis completed successfully', response.data)

    def test_09_api_check_endpoint(self):
        """Verify JSON API endpoint functions correctly and validates input."""
        # Validation failure
        res_fail = self.client.post('/api/check', json={'doc1': 'hi', 'doc2': 'hello'})
        self.assertEqual(res_fail.status_code, 400)
        self.assertIn('error', res_fail.get_json())

        # Success
        res_ok = self.client.post('/api/check', json={
            'doc1': 'Distributed consensus algorithms ensure data consistency across multiple server nodes.',
            'doc2': 'Distributed consensus algorithms ensure database consistency across server nodes.'
        })
        self.assertEqual(res_ok.status_code, 200)
        data = res_ok.get_json()
        self.assertIn('similarity_percentage', data)
        self.assertIn('status', data)
        self.assertGreater(data['similarity_percentage'], 50.0)

    def test_10_history_and_deletion(self):
        """Verify history page display, deletion of record, and clear history."""
        # History page
        res = self.client.get('/history')
        self.assertEqual(res.status_code, 200)

        # Clear history
        res_clear = self.client.post('/history/clear', follow_redirects=True)
        self.assertEqual(res_clear.status_code, 200)
        self.assertIn(b'No Check History Found', res_clear.data)

    def test_11_dashboard_page(self):
        """Verify analytics dashboard renders without errors."""
        res = self.client.get('/dashboard')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Analytics & System Intelligence', res.data)

    def test_12_error_404_handling(self):
        """Verify 404 page is rendered for non-existent routes."""
        res = self.client.get('/non_existent_page_12345')
        self.assertEqual(res.status_code, 404)
        self.assertIn(b'Page Not Found', res.data)

if __name__ == '__main__':
    unittest.main()
