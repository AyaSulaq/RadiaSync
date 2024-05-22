import '../styles/globals.css';
import { UserProvider } from '../services/UserContext';

function App({ Component, pageProps }) {
  return (
    <UserProvider>
      <Component {...pageProps} />
    </UserProvider>
  );
}

export default App;
