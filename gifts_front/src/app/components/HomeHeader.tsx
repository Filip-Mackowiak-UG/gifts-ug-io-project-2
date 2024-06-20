import React from "react";
import {Button} from "@/components/ui/button";
import {Moon, Sun} from "lucide-react";
import {useTheme} from "next-themes";

export function HomeHeader() {
    return (
        <header className="flex justify-between items-center p-4">
            <h1 className="text-2xl font-semibold">Welcome to GIFTS</h1>
            <ThemeButton />
        </header>
    );
}

function ThemeButton() {
    const { setTheme, theme,  } = useTheme()

    const onButtonClick = () => {
        if (theme === "dark") {
            setTheme("light");
        } else {
            setTheme("dark");
        }
    }

    return (
        <Button variant="outline" size="icon" onClick={onButtonClick}>
            <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            <span className="sr-only">Toggle theme</span>
        </Button>
    );
}