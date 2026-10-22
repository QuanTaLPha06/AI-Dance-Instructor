import type { Metadata } from "next";
import { Cinzel, Outfit } from "next/font/google";
import "./globals.css";

const cinzel = Cinzel({
  variable: "--font-cinzel",
  subsets: ["latin"],
  weight: ["400", "600", "700", "800", "900"],
});

const outfit = Outfit({
  variable: "--font-outfit",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Kathak AI Guru • कथक नृत्य मुद्रा परिशोधन",
  description: "AI-Powered Indian Classical Kathak Dance Pose Alignment, Kinematic Analysis & Visual Correction",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${cinzel.variable} ${outfit.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col font-sans bg-[#0c0205] text-[#fdf8eb] selection:bg-amber-500/30 selection:text-amber-200">
        {children}
      </body>
    </html>
  );
}
